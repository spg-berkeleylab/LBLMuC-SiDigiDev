#include "TMath.h"
#include "TFile.h"
#include "TTree.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TCanvas.h"
#include "TTreeReader.h"
#include "TTreeReaderValue.h"
#include "TTreeReaderArray.h"
#include <limits>
using namespace std;
using namespace TMath;

std::map<std::string, TH1 *> h;
const float electrons_per_GeV = 1. / 0.00362 * 1e6;
const int nmax = 1000;
const float c_mm_per_ns = 299.792458;

/*
Plots cluster analysis histograms. Providing timeWindow in ns eliminates clusters outside of -timeWindow and +timeWindow.
*/
void plotHistograms(Double_t timeWindow = NULL)
{
  //Init histograms
  h["clus_charge"] = new TH1F("clus_charge", "Cluster charge;charge (e^{-});a.u.", 100, 0.0, 50000.);
  h["hit_charge"] = new TH1F("hit_charge", "Hit Charge;charge (e^{-});a.u.", 100, 0.0, 10000.);
  h["clus_posres_x"] = new TH1F("clus_posres_x", "Cluster position X (reco - true); #Delta x (#mu m);a.u.", 100, -50., 50.);
  h["clus_posres_y"] = new TH1F("clus_posres_y", "Cluster position Y (reco - true); #Delta y (#mu m);a.u.", 100, -50., 50.);
  h["clus_size_x"] = new TH1F("clus_size_x", "Cluster size in X direction; Size (pixels); a.u.", 50, -0.5, 49.5);
  h["clus_size_y"] = new TH1F("clus_size_y", "Cluster size in Y direction; Size (pixels); a.u.", 50, -0.5, 49.5);
  h["clus_size_y_theta"] = new TH2F("clus_sizeY_theta", "Cluster size in Y direction vs #theta; #theta (deg); Size (pixels); a.u.",
                                    140, 20.0, 160.0, 100, -0.5, 10.5);
  h["hits_rem"] = new TH1F("hits_rem", "Percentage Removal (BIB); % of clusters removed; # of events", 100, 0.0, 100.0);

  Int_t ntrh;
  Int_t ntrc;
  Int_t thcidx[nmax];
  Int_t thclen[nmax];
  Int_t tcrp0[nmax];
  Int_t tcrp1[nmax];
  Int_t h2mf[nmax];
  Float_t stedp[nmax];
  Float_t thedp[nmax];
  Float_t tcedp[nmax];
  Double_t thpox[nmax];
  Double_t thpoy[nmax];
  Double_t thpoz[nmax];
  Double_t stpox[nmax];
  Double_t stpoy[nmax];
  Double_t stpoz[nmax];
  Double_t sttim[nmax];

  TFile *myFile = TFile::Open("../ntuple_tracker.root");
  TTree *t = myFile->Get<TTree>("MyLCTuple;1");
  t->SetBranchAddress("ntrh", &ntrh);
  t->SetBranchAddress("ntrc", &ntrc);
  t->SetBranchAddress("thcidx", thcidx);
  t->SetBranchAddress("thclen", thclen);
  t->SetBranchAddress("tcrp0", tcrp0);
  t->SetBranchAddress("tcrp1", tcrp1);
  t->SetBranchAddress("h2mf", h2mf);
  t->SetBranchAddress("stedp", stedp);
  t->SetBranchAddress("thedp", thedp);
  t->SetBranchAddress("tcedp", tcedp);
  t->SetBranchAddress("thpox", thpox);
  t->SetBranchAddress("thpoy", thpoy);
  t->SetBranchAddress("thpoz", thpoz);
  t->SetBranchAddress("stpox", stpox);
  t->SetBranchAddress("stpoy", stpoy);
  t->SetBranchAddress("stpoz", stpoz);
  t->SetBranchAddress("sttim", sttim);
  Int_t nevents = t->GetEntries();

  for (int i = 0; i < nevents; ++i)
  {
    t->GetEntry(i);
    Int_t clustersExcluded = 0;
    for (int clus = 0; clus < ntrh; ++clus)
    {
      Double_t travTime = Sqrt(Sq(thpox[clus]) + Sq(thpoy[clus]) + Sq(thpoz[clus])) / c_mm_per_ns;
      Double_t clusStart = sttim[clus] - travTime;
      if (timeWindow && (clusStart > timeWindow || clusStart < -timeWindow)) { 
          clustersExcluded++;
          continue; 
      }
      Int_t truth = h2mf[clus];
      Float_t clusCharge = thedp[clus] * electrons_per_GeV;
      h["clus_charge"]->Fill(clusCharge);
      h["clus_posres_x"]->Fill((thpox[clus] - stpox[truth]) * 1e3);
      h["clus_posres_y"]->Fill((thpoy[clus] - stpoy[truth]) * 1e3);

      Int_t minX = INT32_MAX;
      Int_t maxX = INT32_MIN;
      Int_t minY = INT32_MAX;
      Int_t maxY = INT32_MIN;
      for (int hit = thcidx[clus]; hit < thcidx[clus] + thclen[clus]; ++hit)
      {
        h["hit_charge"]->Fill(tcedp[hit]);
        minX = Min(minX, tcrp0[hit]);
        maxX = Max(maxX, tcrp0[hit]);
        minY = Min(minY, tcrp1[hit]);
        maxY = Max(maxY, tcrp1[hit]);
      }
      Int_t sizeX = abs(maxX - minX) + 1;
      Int_t sizeY = abs(maxY - minY) + 1;
      Float_t theta = atan2(stpoy[truth], stpoz[truth]) / Pi() * 180.;
      if (theta < 0)
      {
        theta += 180; //Atan range [-90, 90], want range [0, 180]
      }
      h["clus_size_x"]->Fill(sizeX);
      h["clus_size_y"]->Fill(sizeY);
      h["clus_size_y_theta"]->Fill(theta, sizeY);
    }
    if (ntrh != 0) {
      h["hits_rem"]->Fill(clustersExcluded / ntrh);
    }
  }


  TFile *outputFile = new TFile("analysis2.root", "RECREATE");
  for (auto const &ph : h)
  {
    if(!timeWindow && ph.first == "hits_incl") {
      continue;
    }
    TCanvas *canv = new TCanvas("?", "?", 900, 600);
    ph.second->Draw();
    canv->SaveAs((ph.first + ".png").c_str()); //save pngs of the plots
    ph.second->Write();
  }
}