#include "TH2.h"
#include "TH1.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "THStack.h"
#include "TTree.h"
#include "TTreeReader.h"
#include "TTreeReaderValue.h"
#include "TTreeReaderArray.h"
#include <limits>
#include "TFile.h"
#include "TMath.h"

#include <map>
#include <math.h>


std::map<std::string, TH1*> h;
void initHistograms();

const float electrons_per_keV = 1. / 0.00362;
const std::vector<float> theta_m90_ranges = {0.0, 30.0, 50.0, 70.0}; // |theta - pi/2| (deg)
const std::vector<std::vector<float>> theta_m90_sizeCut       = { {2, 3, 6, 999},
                                                                  {2, 3, 4, 999},
                                                                  {2, 3, 3, 999},
                                                                  {3, 3, 4, 999} }; //[layer][theta_range], last bin is overflow
const std::vector<std::vector<float>> theta_m90_sizeCut_tight = { {2, 3, 5, 5},
                                                                  {2, 3, 4, 5},
                                                                  {2, 3, 3, 3},
                                                                  {2, 2, 2, 3}}; //[layer][theta_range], last bin is overflow

//Input branches
void initBranches(TTree *t);
const int nmax = 1000000; //clusters
const int nmaxh = 10000000; //hits
Int_t ntrh; // number of clusters
Int_t ntrc; // number of hits
Int_t thci0[nmax]; //cellID0
Int_t thci1[nmax]; //cellID1
Int_t thcidx[nmax]; //starting index of hits in the cluster
Int_t thclen[nmax]; //number of hits in the cluster
Int_t tcrp0[nmaxh];
Int_t tcrp1[nmaxh];
Int_t h2mf[nmax];
Int_t h2mt[nmax];
Float_t stedp[nmax];
Float_t thedp[nmax];
Float_t tcedp[nmaxh];
Double_t thpox[nmax];
Double_t thpoy[nmax];
Double_t thpoz[nmax];
Float_t thplen[nmax];
Float_t thtim[nmax];
Double_t stpox[nmax];
Double_t stpoy[nmax];
Double_t stpoz[nmax];
Float_t stmox[nmax];
Float_t stmoy[nmax];
Float_t stmoz[nmax];
//Variables for feature collection
void initOutputFeatures(TTree *t);
Int_t clszx;
Int_t clszy;
Float_t clch;
Float_t clthe;
Float_t clpor;

//Feature Layer selects the nth pair of layers for feature analysis.
void ClusterProperties(std::string inputFile="ntuple_tracker.root", int featureLayer=0)
{

  TFile *fin = TFile::Open(inputFile.c_str());
  if (!fin) {
    std::cout << "Error opening file: " << inputFile << std::endl;
    return;
  }

  TTree *tin = fin->Get<TTree>("MyLCTuple");
  
  initBranches(tin);

  //style
  gStyle->SetOptStat(0000);
  gStyle->SetOptFit(0000);

  //Init histograms
  initHistograms();

  //Init TTree for feature collection
  TFile *featureFile = TFile::Open("features.root", "RECREATE");
  TTree *features = new TTree("FeaturesTree","Features for ML Analysis");
  initOutputFeatures(features);

  float worst_cut = FLT_MAX;
  float worst_cut_loose = FLT_MAX;
  size_t numClusters = 0;

  Long64_t nentries = tin->GetEntries();

  Long64_t nbytes = 0, nb = 0;
  for (Long64_t jentry=0; jentry<nentries;jentry++) {
      tin->GetEntry(jentry);

      int numClusters_size_cut_event = 0;
      int numClusters_size_cut_event_loose = 0;
      for (size_t ic = 0; ic < ntrh; ++ic) {
        //cluster properties
        size_t g4_idx = h2mt[ic]; // note: 1-1 association!
        Int_t side = (thci0[ic] >> 5) & 0b11;
        Int_t layer = (thci0[ic] >> 7) & 0b111111;

        float deltaE = (thedp[ic] - stedp[g4_idx]) * 1e6; // GeV -> keV
        h["clus_deltaE"]->Fill(deltaE);
        float charge = thedp[ic] * 1e6 * electrons_per_keV;
        h["clus_charge"]->Fill(charge);
        h["clus_charge_path"]->Fill(charge, thplen[ic] * 1e3); //path length in um

        float tof = thtim[ic] - std::sqrt(thpox[ic]*thpox[ic]+thpoy[ic]*thpoy[ic]+thpoz[ic]*thpoz[ic])/300.; // c = 300 mm / ns
        h["clus_time"]->Fill(tof);

        //note TODO: project along u,v directions (or just store in u[2], v[2] the position and incidence angle for the two axes!
        h["clus_posres_x"]->Fill((thpox[ic]-stpox[g4_idx]) * 1e3); //mm -> um
        h["clus_posres_y"]->Fill((thpoy[ic]-stpoy[g4_idx]) * 1e3); //mm -> um
        h["clus_posres_z"]->Fill((thpoz[ic]-stpoz[g4_idx]) * 1e3); //mm -> um
        h["clus_pos_z_r"]->Fill(thpoz[ic], std::sqrt(thpox[ic]*thpox[ic]+thpoy[ic]*thpoy[ic]));
        h["clus_truepos_z_r"]->Fill(stpoz[g4_idx], std::sqrt(stpox[g4_idx]*stpox[g4_idx]+stpoy[g4_idx]*stpoy[g4_idx]));
        h["clus_truereco_pos_z"]->Fill(thpoz[ic], stpoz[g4_idx]);

        //hits properties
        int minX=999999; int maxX=-999999; int minY=999999; int maxY=-999999;
        for (size_t ih = thcidx[ic]; ih < thcidx[ic]+thclen[ic]; ++ih) {
          h["hit_charge"]->Fill(tcedp[ih]);
          //store min/max local position (units of pixels)
          if (tcrp0[ih] > maxX) maxX = tcrp0[ih];
          if (tcrp1[ih] > maxY) maxY = tcrp1[ih];          
          if (tcrp0[ih] < minX) minX = tcrp0[ih];
          if (tcrp1[ih] < minY) minY = tcrp1[ih];          
        }

        //cluster shape
        int size_x = std::abs(maxX - minX) + 1;
        int size_y = std::abs(maxY - minY) + 1;
        float theta = std::atan(stmoy[g4_idx] / stmoz[g4_idx]);
        float clus_theta = std::atan(thpoy[ic]/thpoz[ic]);
        //std::cout << "Theta MC = " << theta << ", Theta Cluster = " << clus_theta << ", minX= " << minX << ", maxX= " << maxX << ", minY=" << minY << ", maxY=" << maxY << std::endl;       
        if (clus_theta < 0) clus_theta = TMath::Pi() + clus_theta; // put it into [0, pi[ range
        if (theta < 0) theta = TMath::Pi() + theta; // put it into [0, pi[ range
        float theta_m90_deg = clus_theta / TMath::Pi() * 180.0 - 90.0; // theta - 90.0 (deg)
        h["clus_size_x"]->Fill(size_x);        
        h["clus_size_y"]->Fill(size_y);
        h["clus_size_y_theta"]->Fill(clus_theta / TMath::Pi() * 180.0, size_y);
        int layerIdx = std::floor(layer / 2);
        int thetaIdx = -1;
        for (int jj=theta_m90_ranges.size()-1; jj >= 0; --jj)
          if (std::abs(theta_m90_deg) >= theta_m90_ranges[jj]) {
            thetaIdx = jj;
            break;
          }
        std::string h_name = "clus_size_y_theta_dl" + std::to_string(layerIdx);
        if ((layerIdx >= 0) and (layerIdx <= 3)) {
          h[h_name]->Fill(clus_theta / TMath::Pi() * 180.0, size_y);
          if ((thetaIdx < theta_m90_ranges.size()) and (thetaIdx >= 0)) {
            h_name = h_name + "_tr" + std::to_string(thetaIdx);
            h[h_name]->Fill(size_y);
          } else {
            std::cout << "theta out of range: " << theta_m90_deg << ", thetaIdx = " << thetaIdx << ", layerIdx = " << layerIdx << std::endl;
          }
        } else {
          std::cout << "Layer out of range: " << theta_m90_deg << ", thetaIdx = " << thetaIdx << ", layerIdx = " << layerIdx << std::endl;
          continue;
        }
        
        
        //test hit filtering cuts
        float theta_m90 = std::abs(clus_theta - TMath::Pi()/2);
        int max_sizeY=999;
        if (thetaIdx >= 0) max_sizeY = theta_m90_sizeCut_tight[layerIdx][thetaIdx];
        //if (theta_m90 < 0.52359878) max_sizeY = 2; //30 deg
        //else if (theta_m90 < 0.87266463) max_sizeY = 3; //50 deg
        //else if (theta_m90 < 1.2217305) max_sizeY = 3; //70 deg
        int max_sizeY_loose=999;
        if (thetaIdx >= 0) max_sizeY_loose = theta_m90_sizeCut[layerIdx][thetaIdx];
        //if (theta_m90 < 0.52359878) max_sizeY_loose = 3;
        //else if (theta_m90 < 0.87266463) max_sizeY_loose = 4;
        //else if (theta_m90 < 1.2217305) max_sizeY_loose = 6;

        numClusters++;
        float thpor = std::sqrt(thpox[ic]*thpox[ic]+thpoy[ic]*thpoy[ic]);
        if (size_y <= max_sizeY) { 
          numClusters_size_cut_event++;
          h["clus_kept_xy"]->Fill(thpox[ic], thpoy[ic]);
          h["clus_kept_rz"]->Fill(thpoz[ic], thpor);
        } else {
          h["clus_lost_xy"]->Fill(thpox[ic], thpoy[ic]);
          h["clus_lost_rz"]->Fill(thpoz[ic], thpor);
        }
        if (size_y <= max_sizeY_loose) {
          h["clus_kept_xy_loose"]->Fill(thpox[ic], thpoy[ic]);
          h["clus_kept_rz_loose"]->Fill(thpoz[ic], thpor);
          numClusters_size_cut_event_loose++;
        } else {
          h["clus_lost_xy_loose"]->Fill(thpox[ic], thpoy[ic]);
          h["clus_lost_rz_loose"]->Fill(thpoz[ic], thpor);
        }
        
        //Fill the TTree Features
        if (layer == featureLayer) {
          clszx = size_x;
          clszy = size_y;
          clch = charge;
          clthe = clus_theta;
          clpor = thpor;
          features->Fill();
        }
      } // loop over clusters
      if(ntrh > 0) {
        h["clus_kept"]->Fill(100 * (float)numClusters_size_cut_event / ntrh);
        h["clus_kept_loose"]->Fill(100 * (float)numClusters_size_cut_event_loose / ntrh);
      }
  } // loop over events

  //Save and Close feature outFile:
  featureFile->Write();
  featureFile->Close();
  //Save histograms
  TFile *outROOT = TFile::Open("analysis.root", "RECREATE");
  for (auto ih : h) {
    ih.second->Write();
  }

//Efficiency is average of clus_kept and clus_kept_loose histograms.
  std::cout << "Cluster filter: " << std::endl;
  std::cout << "Tight cut: " << h["clus_kept"]->GetMean() << std::endl;
  std::cout << "Loose cut: " << h["clus_kept_loose"]->GetMean() << std::endl;
}

/* Initializes the branches that are pulled from during analysis. */
void initBranches(TTree *t) 
{
  t->SetBranchAddress("ntrh", &ntrh);
  t->SetBranchAddress("ntrc", &ntrc); 
  t->SetBranchAddress("thci0", thci0);
  t->SetBranchAddress("thci1", thci1);
  t->SetBranchAddress("thcidx", thcidx);
  t->SetBranchAddress("thclen", thclen);
  t->SetBranchAddress("tcrp0", tcrp0);
  t->SetBranchAddress("tcrp1", tcrp1);
  t->SetBranchAddress("h2mf", h2mf);
  t->SetBranchAddress("h2mt", h2mt);  
  t->SetBranchAddress("stedp", stedp);
  t->SetBranchAddress("thedp", thedp);
  t->SetBranchAddress("tcedp", tcedp);
  t->SetBranchAddress("thpox", thpox);
  t->SetBranchAddress("thpoy", thpoy);
  t->SetBranchAddress("thpoz", thpoz);
  t->SetBranchAddress("thplen", thplen);
  t->SetBranchAddress("thtim", thtim);    
  t->SetBranchAddress("stpox", stpox);
  t->SetBranchAddress("stpoy", stpoy);
  t->SetBranchAddress("stpoz", stpoz);
  t->SetBranchAddress("stmox", stmox);
  t->SetBranchAddress("stmoy", stmoy);
  t->SetBranchAddress("stmoz", stmoz);
}

/* Initializes histograms on the global histogram map which stores them. */
void initHistograms()
{
  h["clus_charge"] = new TH1F("clus_charge", "Cluster charge;charge (e^{-});a.u.",
                              100, 0.0, 50000. );   

  h["clus_charge_path"] = new TH2F("clus_charge_path", "Cluster charge vs true path length;charge (e^{-}); path length (#mu m);a.u.",
                              100, 0.0, 50000., 500, 50.0, 550. );   

  h["clus_time"] = new TH1F("clus_time", "Cluster time (TOF #beta=1 corrected); Time (ns); a.u.",
                              130, -0.3, 1. );
  h["clus_deltaE"] = new TH1F("clus_deltaE", "Cluster (reco - true) energy; #Delta E (keV); a.u.",
                              200, -100, 100. );
  h["clus_posres_x"] = new TH1F("clus_posres_x", "Cluster position X (reco - true); #Delta x (#mu m);a.u.",
                              100, -50., 50.);
  h["clus_posres_y"] = new TH1F("clus_posres_y", "Cluster position Y (reco - true); #Delta y (#mu m);a.u.",
                              100, -50., 50.);
  h["clus_posres_z"] = new TH1F("clus_posres_z", "Cluster position Z (reco - true); #Delta z (#mu m);a.u.",
                              100, -50., 50.);
  h["clus_pos_z_r"] = new TH2F("clus_pos_r_z", "Cluster position in R-Z; Z (mm); R(mm); a.u.",
                              160, -80.0, 80.0, 110, 0.0, 110.);
  h["clus_truepos_z_r"] = new TH2F("clus_truepos_r_z", "True Cluster position in R-Z; Z (mm); R(mm); a.u.",
                              160, -80.0, 80.0, 110, 0.0, 110.);
  h["clus_truereco_pos_z"] = new TH2F("clus_truereco_pos_z", "Reco vs True Cluster position in Z; Reco Z (mm); True Z (mm); a.u.",
                              160, -80.0, 80.0, 160, -80.0, 80.);
  h["clus_size_x"] = new TH1F("clus_size_x", "Cluster size in X direction; Size (pixels); a.u.",
                              50, -0.5, 49.5);
  h["clus_size_y"] = new TH1F("clus_size_y", "Cluster size in Y direction; Size (pixels); a.u.",
                              50, -0.5, 49.5);
  h["clus_size_y_theta"] = new TH2F("clus_sizeY_theta", "Cluster size in Y direction vs #theta; #theta (deg); Size (pixels); a.u.",
                              140, 20.0, 160.0, 100, -0.5, 49.5 );
  //cluster size Y as for different layers (2D) and different layers and eta ranges (1D)
  for (int layer_range=0; layer_range < 4; layer_range++) {
    std::string h_name, h_title;
    h_name = "clus_size_y_theta_dl" + std::to_string(layer_range);
    h_title = "Cluster size in Y direction vs #theta (double-layer #" + std::to_string(layer_range) + "); #theta (deg); Size (pixels); a.u.";
    h[h_name] = new TH2F(h_name.c_str(), h_title.c_str(), 140, 20.0, 160.0, 100, -0.5, 49.5 );
    for (int theta_range=0; theta_range < theta_m90_ranges.size(); theta_range++) {
      h_name = "clus_size_y_theta_dl" + std::to_string(layer_range) + "_tr" + std::to_string(theta_range);
      h_title = "Cluster size in Y direction vs #theta (double-layer #" + std::to_string(layer_range) + ", ";
      h_title = h_title + std::to_string(theta_m90_ranges[theta_range]) + " < |#theta - 90deg| ";
      if (theta_range < theta_m90_ranges.size()-1)
        h_title = h_title + "< " + std::to_string(theta_m90_ranges[theta_range+1]);
      h_title = h_title + "); #theta (deg); Size (pixels); a.u.";
      h[h_name] = new TH1F(h_name.c_str(), h_title.c_str(), 100, -0.5, 49.5);
    }
  }
  h["hit_charge"] = new TH1F("hit_charge", "Hit Charge;charge (e^{-});a.u.",
                              100, 0.0, 10000. );
  h["clus_kept_loose"] = new TH1F("clus_kept_loose", "Percentage Kept (Loose); % of clusters kept; # of events", 
                              100, -0.5, 100.5);
  h["clus_kept"] = new TH1F("clus_kept", "Percentage Kept (Tight); % of clusters kept; # of events", 
                              100, -0.5, 100.5);
  h["clus_kept_xy"] = new TH2F("clus_kept_xy", "Reco cluster position of kept (tight); Reco X (mm); Reco Y (mm)", 
                              320, -160.0, 160.0, 320, -160.0, 160.0);
  h["clus_kept_rz"] = new TH2F("clus_kept_rz", "Reco cluster position of kept (tight); Reco Z (mm); Reco R (mm)", 
                              160, -80.0, 80.0, 110, 0.0, 110.);
  h["clus_lost_xy"] = new TH2F("clus_lost_xy", "Reco cluster position of lost (tight); Reco X (mm); Reco Y (mm)", 
                              320, -160.0, 160.0, 320, -160.0, 160.0);
  h["clus_lost_rz"] = new TH2F("clus_lost_rz", "Reco cluster position of lost (tight); Reco Z (mm); Reco R (mm)", 
                              160, -80.0, 80.0, 110, 0.0, 110.);
  h["clus_kept_xy_loose"] = new TH2F("clus_kept_xy_loose", "Reco cluster position X-Y of kept (loose); Reco X (mm); Reco Y (mm)", 
                              320, -160.0, 160.0, 320, -160.0, 160.0);
  h["clus_kept_rz_loose"] = new TH2F("clus_kept_rz_loose", "Reco cluster position R-Z of kept (loose); Reco Z (mm); Reco R (mm);", 
                              160, -80.0, 80.0, 110, 0.0, 110.);
  h["clus_lost_xy_loose"] = new TH2F("clus_lost_xy_loose", "Reco cluster position X-Y of lost (loose); Reco X (mm); Reco Y (mm)", 
                              320, -160.0, 160.0, 320, -160.0, 160.0);
  h["clus_lost_rz_loose"] = new TH2F("clus_lost_rz_loose", "Reco cluster position R-Z of lost (loose); Reco Z (mm); Reco R(mm)", 
                              160, -80.0, 80.0, 110, 0.0, 110.);
}

/* Initializes the branches for a tree t to hold features needed for ML classification.
 * The clusters in each event are flattened into one array for each feature.
 * Currently, the following branches are made:
 *  - Cluster size X "clszX"
 *  - Cluster size Y "clszY"
 *  - Cluster Charge "clch"
 *  - Cluster Theta (Reconstructed) "clthe"
 *  - Cluster R (Reconstructed) "clpor"
 *  Future ideas:
 *  - Cluster time delta
 */
void initOutputFeatures(TTree *t) 
{
  t->Branch("clszx", &clszx);
  t->Branch("clszy", &clszy);
  t->Branch("clch", &clch);
  t->Branch("clthe", &clthe);
  t->Branch("clpor", &clpor);
}
