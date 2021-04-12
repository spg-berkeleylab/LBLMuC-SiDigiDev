#include "TH2.h"
#include "TStyle.h"
#include "TCanvas.h"
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

const float electrons_per_keV = 1. / 0.00362;

//Input branches
void initBranches(TTree *t);
const int nmax = 100000;
Int_t ntrh;
Int_t ntrc;
Int_t thcidx[nmax];
Int_t thclen[nmax];
Int_t tcrp0[nmax];
Int_t tcrp1[nmax];
Int_t h2mf[nmax];
Int_t h2mt[nmax];
Float_t stedp[nmax];
Float_t thedp[nmax];
Float_t tcedp[nmax];
Double_t thpox[nmax];
Double_t thpoy[nmax];
Double_t thpoz[nmax];
Float_t thplen[nmax];
Double_t stpox[nmax];
Double_t stpoy[nmax];
Double_t stpoz[nmax];
Float_t stmox[nmax];
Float_t stmoy[nmax];
Float_t stmoz[nmax];

void ClusterProperties(std::string inputFile="ntuple_tracker.root")
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
  h["clus_charge"] = new TH1F("clus_charge", "Cluster charge;charge (e^{-});a.u.",
                              100, 0.0, 50000. );   

  h["clus_charge_path"] = new TH2F("clus_charge_path", "Cluster charge vs true path length;charge (e^{-}); path length (#mu m);a.u.",
                                    100, 0.0, 50000., 500, 50.0, 550. );   

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
  h["hit_charge"] = new TH1F("hit_charge", "Hit Charge;charge (e^{-});a.u.",
                              100, 0.0, 10000. );

  size_t numClusters_size_cut = 0;
  size_t numClusters_size_cut_loose = 0;
  size_t numClusters = 0;

  Long64_t nentries = tin->GetEntries();

  Long64_t nbytes = 0, nb = 0;
  for (Long64_t jentry=0; jentry<nentries;jentry++) {
      tin->GetEntry(jentry);

      for (size_t ic = 0; ic < ntrh; ++ic) {
        //cluster properties
        size_t g4_idx = h2mt[ic]; // note: 1-1 association!

        float deltaE = (thedp[ic] - stedp[g4_idx]) * 1e6; // GeV -> keV
        h["clus_deltaE"]->Fill(deltaE);
        float charge = thedp[ic] * 1e6 * electrons_per_keV;
        h["clus_charge"]->Fill(charge);
        h["clus_charge_path"]->Fill(charge, thplen[ic] * 1e3); //path length in um

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
        if (theta < 0) theta = TMath::Pi() + theta; // put it into [0, pi[ range
        h["clus_size_x"]->Fill(size_x);        
        h["clus_size_y"]->Fill(size_y);
        h["clus_size_y_theta"]->Fill(theta / TMath::Pi() * 180.0, size_y);

        //test hit filtering cuts
        float theta_m90 = std::abs(theta - TMath::Pi()/2);
        int max_sizeY=999;
        if (theta_m90 < 0.52359878) max_sizeY = 2; //30 deg
        else if (theta_m90 < 0.87266463) max_sizeY = 3; //50 deg
        else if (theta_m90 < 1.2217305) max_sizeY = 3; //70 deg
        int max_sizeY_loose=999;
        if (theta_m90 < 0.52359878) max_sizeY_loose = 3;
        else if (theta_m90 < 0.87266463) max_sizeY_loose = 4;
        else if (theta_m90 < 1.2217305) max_sizeY_loose = 6;

        numClusters++;
        if (size_y < max_sizeY) numClusters_size_cut++;
        if (size_y < max_sizeY_loose) numClusters_size_cut_loose++;        
        
      } // loop over clusters
      
  } // loop over events

  //Save histograms
  TFile *outROOT = TFile::Open("analysis.root", "RECREATE");
  for (auto ih : h) {
    ih.second->Write();
  }

  std::cout << "Cluster filter efficiency: " << std::endl;
  std::cout << "Tight cut: " << (float)numClusters_size_cut / numClusters << std::endl;
  std::cout << "Loose cut: " << (float)numClusters_size_cut_loose / numClusters << std::endl;
}

void initBranches(TTree *t) 
{
  t->SetBranchAddress("ntrh", &ntrh);
  t->SetBranchAddress("ntrc", &ntrc);
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
  t->SetBranchAddress("stpox", stpox);
  t->SetBranchAddress("stpoy", stpoy);
  t->SetBranchAddress("stpoz", stpoz);
  t->SetBranchAddress("stmox", stmox);
  t->SetBranchAddress("stmoy", stmoy);
  t->SetBranchAddress("stmoz", stmoz);
}
