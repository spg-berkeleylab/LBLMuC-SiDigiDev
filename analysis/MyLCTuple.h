//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Fri Mar 19 19:01:14 2021 by ROOT version 6.22/06
// from TTree MyLCTuple/columnwise ntuple with LCIO data
// found on file: ntuple_tracker.root
//////////////////////////////////////////////////////////

#ifndef MyLCTuple_h
#define MyLCTuple_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>

// Header file for the classes stored in the TTree if any.
#include "vector"

class MyLCTuple {
public :
   TTree          *fChain;   //!pointer to the analyzed TTree or TChain
   Int_t           fCurrent; //!current Tree number in a TChain

// Fixed size dimensions of array or collections stored in the TTree if any.

   // Declaration of leaf types
   Int_t           evevt;
   Int_t           evrun;
   Float_t         evwgt;
   Long64_t        evtim;
   Float_t         evsig;
   Float_t         evene;
   Float_t         evpoe;
   Float_t         evpop;
   Int_t           evnch;
   Char_t          evpro[1];   //[evnch]
   Int_t           nj;
   Float_t         jmox[1];   //[njet]
   Float_t         jmoy[1];   //[njet]
   Float_t         jmoz[1];   //[njet]
   Float_t         jmas[1];   //[njet]
   Float_t         jene[1];   //[njet]
   Float_t         jcha[1];   //[njet]
   Float_t         jcov0[1];   //[njet]
   Float_t         jcov1[1];   //[njet]
   Float_t         jcov2[1];   //[njet]
   Float_t         jcov3[1];   //[njet]
   Float_t         jcov4[1];   //[njet]
   Float_t         jcov5[1];   //[njet]
   Float_t         jcov6[1];   //[njet]
   Float_t         jcov7[1];   //[njet]
   Float_t         jcov8[1];   //[njet]
   Float_t         jcov9[1];   //[njet]
   Float_t         jevis;
   Float_t         jPxvis;
   Float_t         jPyvis;
   Float_t         jPzvis;
   Float_t         jmom[1];   //[njet]
   Float_t         jcost[1];   //[njet]
   Float_t         jcosTheta;
   Float_t         jTheta;
   Float_t         jPtvis;
   Float_t         jmvis;
   Float_t         jmmax;
   Float_t         jEmiss;
   Float_t         jMmissq;
   Float_t         jMmiss;
   Int_t           trnpar;
   vector<string>  *trpana;
   vector<string>  *trpaor;
   Int_t           trpaod[1];   //[trnpar]
   Int_t           trpain[1];   //[trnpar]
   Int_t           trpaiv[1][50];   //[trnpar]
   Int_t           trpafn[1];   //[trnpar]
   Float_t         trpafv[1][50];   //[trnpar]
   Int_t           trpasn[1];   //[trnpar]
   Int_t           ntrk;
   Int_t           trori[1];   //[ntrk]
   Int_t           trtyp[1];   //[ntrk]
   Float_t         trch2[1];   //[ntrk]
   Int_t           trndf[1];   //[ntrk]
   Float_t         tredx[1];   //[ntrk]
   Float_t         trede[1];   //[ntrk]
   Float_t         trrih[1];   //[ntrk]
   Int_t           trthn[1];   //[ntrk]
   Int_t           trthi[1][50];   //[ntrk]
   Int_t           trshn[1][12];   //[ntrk]
   Int_t           trnts[1];   //[ntrk]
   Int_t           trfts[1];   //[ntrk]
   Int_t           trsip[1];   //[ntrk]
   Int_t           trsfh[1];   //[ntrk]
   Int_t           trslh[1];   //[ntrk]
   Int_t           trsca[1];   //[ntrk]
   Int_t           ntrst;
   Int_t           tsloc[1];   //[ntrst]
   Float_t         tsdze[1];   //[ntrst]
   Float_t         tsphi[1];   //[ntrst]
   Float_t         tsome[1];   //[ntrst]
   Float_t         tszze[1];   //[ntrst]
   Float_t         tstnl[1];   //[ntrst]
   Float_t         tscov[1][15];   //[ntrst]
   Float_t         tsrpx[1];   //[ntrst]
   Float_t         tsrpy[1];   //[ntrst]
   Float_t         tsrpz[1];   //[ntrst]
   Int_t           stnpar;
   vector<string>  *stpana;
   vector<string>  *stpaor;
   Int_t           stpaod[1];   //[stnpar]
   Int_t           stpain[1];   //[stnpar]
   Int_t           stpaiv[1][50];   //[stnpar]
   Int_t           stpafn[1];   //[stnpar]
   Float_t         stpafv[1][50];   //[stnpar]
   Int_t           stpasn[1];   //[stnpar]
   Int_t           nsth;
   Int_t           stori[218];   //[nsth]
   Int_t           stci0[218];   //[nsth]
   Int_t           stci1[218];   //[nsth]
   Double_t        stpox[218];   //[nsth]
   Double_t        stpoy[218];   //[nsth]
   Double_t        stpoz[218];   //[nsth]
   Float_t         stedp[218];   //[nsth]
   Float_t         sttim[218];   //[nsth]
   Float_t         stmox[218];   //[nsth]
   Float_t         stmoy[218];   //[nsth]
   Float_t         stmoz[218];   //[nsth]
   Float_t         stptl[218];   //[nsth]
   Int_t           stmcp[218];   //[nsth]
   Int_t           ntrh;
   Int_t           thori[211];   //[ntrh]
   Int_t           thci0[211];   //[ntrh]
   Int_t           thci1[211];   //[ntrh]
   Double_t        thpox[211];   //[ntrh]
   Double_t        thpoy[211];   //[ntrh]
   Double_t        thpoz[211];   //[ntrh]
   Float_t         thedp[211];   //[ntrh]
   Float_t         thtim[211];   //[ntrh]
   Float_t         thcov[211][6];   //[ntrh]
   Float_t         thtyp[211];   //[ntrh]
   Float_t         thqua[211];   //[ntrh]
   Float_t         thede[211];   //[ntrh]
   Float_t         thplen[211];   //[ntrh]
   Int_t           thsrc[211];   //[ntrh]
   Int_t           thcidx[211];   //[ntrh]
   Int_t           thclen[211];   //[ntrh]
   Int_t           ntrc;
   Float_t         tcedp[4391];   //[ntrc]
   Float_t         tctim[4391];   //[ntrc]
   Int_t           tcrp0[4391];   //[ntrc]
   Int_t           tcrp1[4391];   //[ntrc]
   Int_t           vtnpar;
   vector<string>  *vtpana;
   vector<string>  *vtpaor;
   Int_t           vtpaod[1];   //[vtnpar]
   Int_t           vtpain[1];   //[vtnpar]
   Int_t           vtpaiv[1][50];   //[vtnpar]
   Int_t           vtpafn[1];   //[vtnpar]
   Float_t         vtpafv[1][50];   //[vtnpar]
   Int_t           vtpasn[1];   //[vtnpar]
   Int_t           nvt;
   Int_t           vtori[1];   //[nvt]
   Int_t           vtpri[1];   //[nvt]
   Int_t           vtrpl[1];   //[nvt]
   Char_t          vttyp[1];   //[nvt]
   Float_t         vtxxx[1];   //[nvt]
   Float_t         vtyyy[1];   //[nvt]
   Float_t         vtzzz[1];   //[nvt]
   Float_t         vtchi[1];   //[nvt]
   Float_t         vtprb[1];   //[nvt]
   Float_t         vtcov[1][6];   //[nvt]
   Float_t         vtpar[1][6];   //[nvt]
   Int_t           r2mnrel;
   Int_t           r2mf[1];   //[r2mnrel]
   Int_t           r2mt[1];   //[r2mnrel]
   Float_t         r2mw[1];   //[r2mnrel]
   Int_t           r2tnrel;
   Int_t           r2tf[1];   //[r2tnrel]
   Int_t           r2tt[1];   //[r2tnrel]
   Float_t         r2tw[1];   //[r2tnrel]
   Int_t           h2mnrel;
   Int_t           h2mf[211];   //[h2mnrel]
   Int_t           h2mt[211];   //[h2mnrel]
   Float_t         h2mw[211];   //[h2mnrel]

   // List of branches
   TBranch        *b_evevt;   //!
   TBranch        *b_evrun;   //!
   TBranch        *b_evwgt;   //!
   TBranch        *b_evtim;   //!
   TBranch        *b_evsig;   //!
   TBranch        *b_evene;   //!
   TBranch        *b_evpoe;   //!
   TBranch        *b_evpop;   //!
   TBranch        *b_evnch;   //!
   TBranch        *b_evpro;   //!
   TBranch        *b_njet;   //!
   TBranch        *b_jmox;   //!
   TBranch        *b_jmoy;   //!
   TBranch        *b_jmoz;   //!
   TBranch        *b_jmas;   //!
   TBranch        *b_jene;   //!
   TBranch        *b_jcha;   //!
   TBranch        *b_jcov0;   //!
   TBranch        *b_jcov1;   //!
   TBranch        *b_jcov2;   //!
   TBranch        *b_jcov3;   //!
   TBranch        *b_jcov4;   //!
   TBranch        *b_jcov5;   //!
   TBranch        *b_jcov6;   //!
   TBranch        *b_jcov7;   //!
   TBranch        *b_jcov8;   //!
   TBranch        *b_jcov9;   //!
   TBranch        *b_jevis;   //!
   TBranch        *b_jPxvis;   //!
   TBranch        *b_jPyvis;   //!
   TBranch        *b_jPzvis;   //!
   TBranch        *b_jmom;   //!
   TBranch        *b_jcost;   //!
   TBranch        *b_jcosTheta;   //!
   TBranch        *b_jTheta;   //!
   TBranch        *b_jPtvis;   //!
   TBranch        *b_jmvis;   //!
   TBranch        *b_jmmax;   //!
   TBranch        *b_jEmiss;   //!
   TBranch        *b_jMmissq;   //!
   TBranch        *b_jMmiss;   //!
   TBranch        *b_trnpar;   //!
   TBranch        *b_trpana;   //!
   TBranch        *b_trpaor;   //!
   TBranch        *b_trpaod;   //!
   TBranch        *b_trpain;   //!
   TBranch        *b_trpaiv;   //!
   TBranch        *b_trpafn;   //!
   TBranch        *b_trpafv;   //!
   TBranch        *b_trpasn;   //!
   TBranch        *b_ntrk;   //!
   TBranch        *b_trori;   //!
   TBranch        *b_trtyp;   //!
   TBranch        *b_trch2;   //!
   TBranch        *b_trndf;   //!
   TBranch        *b_tredx;   //!
   TBranch        *b_trede;   //!
   TBranch        *b_trrih;   //!
   TBranch        *b_trthn;   //!
   TBranch        *b_trthi;   //!
   TBranch        *b_trshn;   //!
   TBranch        *b_trnts;   //!
   TBranch        *b_trfts;   //!
   TBranch        *b_trsip;   //!
   TBranch        *b_trsfh;   //!
   TBranch        *b_trslh;   //!
   TBranch        *b_trsca;   //!
   TBranch        *b_ntrst;   //!
   TBranch        *b_tsloc;   //!
   TBranch        *b_tsdze;   //!
   TBranch        *b_tsphi;   //!
   TBranch        *b_tsome;   //!
   TBranch        *b_tszze;   //!
   TBranch        *b_tstnl;   //!
   TBranch        *b_tscov;   //!
   TBranch        *b_tsrpx;   //!
   TBranch        *b_tsrpy;   //!
   TBranch        *b_tsrpz;   //!
   TBranch        *b_stnpar;   //!
   TBranch        *b_stpana;   //!
   TBranch        *b_stpaor;   //!
   TBranch        *b_stpaod;   //!
   TBranch        *b_stpain;   //!
   TBranch        *b_stpaiv;   //!
   TBranch        *b_stpafn;   //!
   TBranch        *b_stpafv;   //!
   TBranch        *b_stpasn;   //!
   TBranch        *b_nsth;   //!
   TBranch        *b_stori;   //!
   TBranch        *b_stci0;   //!
   TBranch        *b_stci1;   //!
   TBranch        *b_stpox;   //!
   TBranch        *b_stpoy;   //!
   TBranch        *b_stpoz;   //!
   TBranch        *b_stedp;   //!
   TBranch        *b_sttim;   //!
   TBranch        *b_stmox;   //!
   TBranch        *b_stmoy;   //!
   TBranch        *b_stmoz;   //!
   TBranch        *b_stptl;   //!
   TBranch        *b_stmcp;   //!
   TBranch        *b_ntrh;   //!
   TBranch        *b_thori;   //!
   TBranch        *b_thci0;   //!
   TBranch        *b_thci1;   //!
   TBranch        *b_thpox;   //!
   TBranch        *b_thpoy;   //!
   TBranch        *b_thpoz;   //!
   TBranch        *b_thedp;   //!
   TBranch        *b_thtim;   //!
   TBranch        *b_thcov;   //!
   TBranch        *b_thtyp;   //!
   TBranch        *b_thqua;   //!
   TBranch        *b_thede;   //!
   TBranch        *b_thplen;   //!
   TBranch        *b_thsrc;   //!
   TBranch        *b_thcidx;   //!
   TBranch        *b_thclen;   //!
   TBranch        *b_ntrc;   //!
   TBranch        *b_tcedp;   //!
   TBranch        *b_tctim;   //!
   TBranch        *b_tcrp0;   //!
   TBranch        *b_tcrp1;   //!
   TBranch        *b_vtnpar;   //!
   TBranch        *b_vtpana;   //!
   TBranch        *b_vtpaor;   //!
   TBranch        *b_vtpaod;   //!
   TBranch        *b_vtpain;   //!
   TBranch        *b_vtpaiv;   //!
   TBranch        *b_vtpafn;   //!
   TBranch        *b_vtpafv;   //!
   TBranch        *b_vtpasn;   //!
   TBranch        *b_nvt;   //!
   TBranch        *b_vtori;   //!
   TBranch        *b_vtpri;   //!
   TBranch        *b_vtrpl;   //!
   TBranch        *b_vttyp;   //!
   TBranch        *b_vtxxx;   //!
   TBranch        *b_vtyyy;   //!
   TBranch        *b_vtzzz;   //!
   TBranch        *b_vtchi;   //!
   TBranch        *b_vtprb;   //!
   TBranch        *b_vtcov;   //!
   TBranch        *b_vtpar;   //!
   TBranch        *b_r2mnrel;   //!
   TBranch        *b_r2mf;   //!
   TBranch        *b_r2mt;   //!
   TBranch        *b_r2mw;   //!
   TBranch        *b_r2tnrel;   //!
   TBranch        *b_r2tf;   //!
   TBranch        *b_r2tt;   //!
   TBranch        *b_r2tw;   //!
   TBranch        *b_h2mnrel;   //!
   TBranch        *b_h2mf;   //!
   TBranch        *b_h2mt;   //!
   TBranch        *b_h2mw;   //!

   MyLCTuple(TTree *tree=0);
   virtual ~MyLCTuple();
   virtual Int_t    Cut(Long64_t entry);
   virtual Int_t    GetEntry(Long64_t entry);
   virtual Long64_t LoadTree(Long64_t entry);
   virtual void     Init(TTree *tree);
   virtual void     Loop();
   virtual Bool_t   Notify();
   virtual void     Show(Long64_t entry = -1);
};

#endif

#ifdef MyLCTuple_cxx
MyLCTuple::MyLCTuple(TTree *tree) : fChain(0) 
{
// if parameter tree is not specified (or zero), connect the file
// used to generate this class and read the Tree.
   if (tree == 0) {
      TFile *f = (TFile*)gROOT->GetListOfFiles()->FindObject("ntuple_tracker.root");
      if (!f || !f->IsOpen()) {
         f = new TFile("ntuple_tracker.root");
      }
      f->GetObject("MyLCTuple",tree);

   }
   Init(tree);
}

MyLCTuple::~MyLCTuple()
{
   if (!fChain) return;
   delete fChain->GetCurrentFile();
}

Int_t MyLCTuple::GetEntry(Long64_t entry)
{
// Read contents of entry.
   if (!fChain) return 0;
   return fChain->GetEntry(entry);
}
Long64_t MyLCTuple::LoadTree(Long64_t entry)
{
// Set the environment to read one entry
   if (!fChain) return -5;
   Long64_t centry = fChain->LoadTree(entry);
   if (centry < 0) return centry;
   if (fChain->GetTreeNumber() != fCurrent) {
      fCurrent = fChain->GetTreeNumber();
      Notify();
   }
   return centry;
}

void MyLCTuple::Init(TTree *tree)
{
   // The Init() function is called when the selector needs to initialize
   // a new tree or chain. Typically here the branch addresses and branch
   // pointers of the tree will be set.
   // It is normally not necessary to make changes to the generated
   // code, but the routine can be extended by the user if needed.
   // Init() will be called many times when running on PROOF
   // (once per file to be processed).

   // Set object pointer
   trpana = 0;
   trpaor = 0;
   stpana = 0;
   stpaor = 0;
   stpana = 0;
   stpaor = 0;
   vtpana = 0;
   vtpaor = 0;
   // Set branch addresses and branch pointers
   if (!tree) return;
   fChain = tree;
   fCurrent = -1;
   fChain->SetMakeClass(1);

   fChain->SetBranchAddress("evevt", &evevt, &b_evevt);
   fChain->SetBranchAddress("evrun", &evrun, &b_evrun);
   fChain->SetBranchAddress("evwgt", &evwgt, &b_evwgt);
   fChain->SetBranchAddress("evtim", &evtim, &b_evtim);
   fChain->SetBranchAddress("evsig", &evsig, &b_evsig);
   fChain->SetBranchAddress("evene", &evene, &b_evene);
   fChain->SetBranchAddress("evpoe", &evpoe, &b_evpoe);
   fChain->SetBranchAddress("evpop", &evpop, &b_evpop);
   fChain->SetBranchAddress("evnch", &evnch, &b_evnch);
   fChain->SetBranchAddress("evpro", &evpro, &b_evpro);
   fChain->SetBranchAddress("nj", &nj, &b_njet);
   fChain->SetBranchAddress("jmox", &jmox, &b_jmox);
   fChain->SetBranchAddress("jmoy", &jmoy, &b_jmoy);
   fChain->SetBranchAddress("jmoz", &jmoz, &b_jmoz);
   fChain->SetBranchAddress("jmas", &jmas, &b_jmas);
   fChain->SetBranchAddress("jene", &jene, &b_jene);
   fChain->SetBranchAddress("jcha", &jcha, &b_jcha);
   fChain->SetBranchAddress("jcov0", &jcov0, &b_jcov0);
   fChain->SetBranchAddress("jcov1", &jcov1, &b_jcov1);
   fChain->SetBranchAddress("jcov2", &jcov2, &b_jcov2);
   fChain->SetBranchAddress("jcov3", &jcov3, &b_jcov3);
   fChain->SetBranchAddress("jcov4", &jcov4, &b_jcov4);
   fChain->SetBranchAddress("jcov5", &jcov5, &b_jcov5);
   fChain->SetBranchAddress("jcov6", &jcov6, &b_jcov6);
   fChain->SetBranchAddress("jcov7", &jcov7, &b_jcov7);
   fChain->SetBranchAddress("jcov8", &jcov8, &b_jcov8);
   fChain->SetBranchAddress("jcov9", &jcov9, &b_jcov9);
   fChain->SetBranchAddress("jevis", &jevis, &b_jevis);
   fChain->SetBranchAddress("jPxvis", &jPxvis, &b_jPxvis);
   fChain->SetBranchAddress("jPyvis", &jPyvis, &b_jPyvis);
   fChain->SetBranchAddress("jPzvis", &jPzvis, &b_jPzvis);
   fChain->SetBranchAddress("jmom", &jmom, &b_jmom);
   fChain->SetBranchAddress("jcost", &jcost, &b_jcost);
   fChain->SetBranchAddress("jcosTheta", &jcosTheta, &b_jcosTheta);
   fChain->SetBranchAddress("jTheta", &jTheta, &b_jTheta);
   fChain->SetBranchAddress("jPtvis", &jPtvis, &b_jPtvis);
   fChain->SetBranchAddress("jmvis", &jmvis, &b_jmvis);
   fChain->SetBranchAddress("jmmax", &jmmax, &b_jmmax);
   fChain->SetBranchAddress("jEmiss", &jEmiss, &b_jEmiss);
   fChain->SetBranchAddress("jMmissq", &jMmissq, &b_jMmissq);
   fChain->SetBranchAddress("jMmiss", &jMmiss, &b_jMmiss);
   fChain->SetBranchAddress("trnpar", &trnpar, &b_trnpar);
   fChain->SetBranchAddress("trpana", &trpana, &b_trpana);
   fChain->SetBranchAddress("trpaor", &trpaor, &b_trpaor);
   fChain->SetBranchAddress("trpaod", &trpaod, &b_trpaod);
   fChain->SetBranchAddress("trpain", &trpain, &b_trpain);
   fChain->SetBranchAddress("trpaiv", &trpaiv, &b_trpaiv);
   fChain->SetBranchAddress("trpafn", &trpafn, &b_trpafn);
   fChain->SetBranchAddress("trpafv", &trpafv, &b_trpafv);
   fChain->SetBranchAddress("trpasn", &trpasn, &b_trpasn);
   fChain->SetBranchAddress("ntrk", &ntrk, &b_ntrk);
   fChain->SetBranchAddress("trori", &trori, &b_trori);
   fChain->SetBranchAddress("trtyp", &trtyp, &b_trtyp);
   fChain->SetBranchAddress("trch2", &trch2, &b_trch2);
   fChain->SetBranchAddress("trndf", &trndf, &b_trndf);
   fChain->SetBranchAddress("tredx", &tredx, &b_tredx);
   fChain->SetBranchAddress("trede", &trede, &b_trede);
   fChain->SetBranchAddress("trrih", &trrih, &b_trrih);
   fChain->SetBranchAddress("trthn", &trthn, &b_trthn);
   fChain->SetBranchAddress("trthi", &trthi, &b_trthi);
   fChain->SetBranchAddress("trshn", &trshn, &b_trshn);
   fChain->SetBranchAddress("trnts", &trnts, &b_trnts);
   fChain->SetBranchAddress("trfts", &trfts, &b_trfts);
   fChain->SetBranchAddress("trsip", &trsip, &b_trsip);
   fChain->SetBranchAddress("trsfh", &trsfh, &b_trsfh);
   fChain->SetBranchAddress("trslh", &trslh, &b_trslh);
   fChain->SetBranchAddress("trsca", &trsca, &b_trsca);
   fChain->SetBranchAddress("ntrst", &ntrst, &b_ntrst);
   fChain->SetBranchAddress("tsloc", &tsloc, &b_tsloc);
   fChain->SetBranchAddress("tsdze", &tsdze, &b_tsdze);
   fChain->SetBranchAddress("tsphi", &tsphi, &b_tsphi);
   fChain->SetBranchAddress("tsome", &tsome, &b_tsome);
   fChain->SetBranchAddress("tszze", &tszze, &b_tszze);
   fChain->SetBranchAddress("tstnl", &tstnl, &b_tstnl);
   fChain->SetBranchAddress("tscov", &tscov, &b_tscov);
   fChain->SetBranchAddress("tsrpx", &tsrpx, &b_tsrpx);
   fChain->SetBranchAddress("tsrpy", &tsrpy, &b_tsrpy);
   fChain->SetBranchAddress("tsrpz", &tsrpz, &b_tsrpz);
   fChain->SetBranchAddress("stnpar", &stnpar, &b_stnpar);
   fChain->SetBranchAddress("stpana", &stpana, &b_stpana);
   fChain->SetBranchAddress("stpaor", &stpaor, &b_stpaor);
   fChain->SetBranchAddress("stpaod", stpaod, &b_stpaod);
   fChain->SetBranchAddress("stpain", stpain, &b_stpain);
   fChain->SetBranchAddress("stpaiv", stpaiv, &b_stpaiv);
   fChain->SetBranchAddress("stpafn", stpafn, &b_stpafn);
   fChain->SetBranchAddress("stpafv", stpafv, &b_stpafv);
   fChain->SetBranchAddress("stpasn", stpasn, &b_stpasn);
   fChain->SetBranchAddress("nsth", &nsth, &b_nsth);
   fChain->SetBranchAddress("stori", stori, &b_stori);
   fChain->SetBranchAddress("stci0", stci0, &b_stci0);
   fChain->SetBranchAddress("stci1", stci1, &b_stci1);
   fChain->SetBranchAddress("stpox", stpox, &b_stpox);
   fChain->SetBranchAddress("stpoy", stpoy, &b_stpoy);
   fChain->SetBranchAddress("stpoz", stpoz, &b_stpoz);
   fChain->SetBranchAddress("stedp", stedp, &b_stedp);
   fChain->SetBranchAddress("sttim", sttim, &b_sttim);
   fChain->SetBranchAddress("stmox", stmox, &b_stmox);
   fChain->SetBranchAddress("stmoy", stmoy, &b_stmoy);
   fChain->SetBranchAddress("stmoz", stmoz, &b_stmoz);
   fChain->SetBranchAddress("stptl", stptl, &b_stptl);
   fChain->SetBranchAddress("stmcp", stmcp, &b_stmcp);
//    fChain->SetBranchAddress("stnpar", &stnpar, &b_stnpar);
//    fChain->SetBranchAddress("stpana", &stpana, &b_stpana);
//    fChain->SetBranchAddress("stpaor", &stpaor, &b_stpaor);
//    fChain->SetBranchAddress("stpaod", stpaod, &b_stpaod);
//    fChain->SetBranchAddress("stpain", stpain, &b_stpain);
//    fChain->SetBranchAddress("stpaiv", stpaiv, &b_stpaiv);
//    fChain->SetBranchAddress("stpafn", stpafn, &b_stpafn);
//    fChain->SetBranchAddress("stpafv", stpafv, &b_stpafv);
//    fChain->SetBranchAddress("stpasn", stpasn, &b_stpasn);
   fChain->SetBranchAddress("ntrh", &ntrh, &b_ntrh);
   fChain->SetBranchAddress("thori", thori, &b_thori);
   fChain->SetBranchAddress("thci0", thci0, &b_thci0);
   fChain->SetBranchAddress("thci1", thci1, &b_thci1);
   fChain->SetBranchAddress("thpox", thpox, &b_thpox);
   fChain->SetBranchAddress("thpoy", thpoy, &b_thpoy);
   fChain->SetBranchAddress("thpoz", thpoz, &b_thpoz);
   fChain->SetBranchAddress("thedp", thedp, &b_thedp);
   fChain->SetBranchAddress("thtim", thtim, &b_thtim);
   fChain->SetBranchAddress("thcov", thcov, &b_thcov);
   fChain->SetBranchAddress("thtyp", thtyp, &b_thtyp);
   fChain->SetBranchAddress("thqua", thqua, &b_thqua);
   fChain->SetBranchAddress("thede", thede, &b_thede);
   fChain->SetBranchAddress("thplen", thplen, &b_thplen);
   fChain->SetBranchAddress("thsrc", thsrc, &b_thsrc);
   fChain->SetBranchAddress("thcidx", thcidx, &b_thcidx);
   fChain->SetBranchAddress("thclen", thclen, &b_thclen);
   fChain->SetBranchAddress("ntrc", &ntrc, &b_ntrc);
   fChain->SetBranchAddress("tcedp", tcedp, &b_tcedp);
   fChain->SetBranchAddress("tctim", tctim, &b_tctim);
   fChain->SetBranchAddress("tcrp0", tcrp0, &b_tcrp0);
   fChain->SetBranchAddress("tcrp1", tcrp1, &b_tcrp1);
   fChain->SetBranchAddress("vtnpar", &vtnpar, &b_vtnpar);
   fChain->SetBranchAddress("vtpana", &vtpana, &b_vtpana);
   fChain->SetBranchAddress("vtpaor", &vtpaor, &b_vtpaor);
   fChain->SetBranchAddress("vtpaod", &vtpaod, &b_vtpaod);
   fChain->SetBranchAddress("vtpain", &vtpain, &b_vtpain);
   fChain->SetBranchAddress("vtpaiv", &vtpaiv, &b_vtpaiv);
   fChain->SetBranchAddress("vtpafn", &vtpafn, &b_vtpafn);
   fChain->SetBranchAddress("vtpafv", &vtpafv, &b_vtpafv);
   fChain->SetBranchAddress("vtpasn", &vtpasn, &b_vtpasn);
   fChain->SetBranchAddress("nvt", &nvt, &b_nvt);
   fChain->SetBranchAddress("vtori", &vtori, &b_vtori);
   fChain->SetBranchAddress("vtpri", &vtpri, &b_vtpri);
   fChain->SetBranchAddress("vtrpl", &vtrpl, &b_vtrpl);
   fChain->SetBranchAddress("vttyp", &vttyp, &b_vttyp);
   fChain->SetBranchAddress("vtxxx", &vtxxx, &b_vtxxx);
   fChain->SetBranchAddress("vtyyy", &vtyyy, &b_vtyyy);
   fChain->SetBranchAddress("vtzzz", &vtzzz, &b_vtzzz);
   fChain->SetBranchAddress("vtchi", &vtchi, &b_vtchi);
   fChain->SetBranchAddress("vtprb", &vtprb, &b_vtprb);
   fChain->SetBranchAddress("vtcov", &vtcov, &b_vtcov);
   fChain->SetBranchAddress("vtpar", &vtpar, &b_vtpar);
   fChain->SetBranchAddress("r2mnrel", &r2mnrel, &b_r2mnrel);
   fChain->SetBranchAddress("r2mf", &r2mf, &b_r2mf);
   fChain->SetBranchAddress("r2mt", &r2mt, &b_r2mt);
   fChain->SetBranchAddress("r2mw", &r2mw, &b_r2mw);
   fChain->SetBranchAddress("r2tnrel", &r2tnrel, &b_r2tnrel);
   fChain->SetBranchAddress("r2tf", &r2tf, &b_r2tf);
   fChain->SetBranchAddress("r2tt", &r2tt, &b_r2tt);
   fChain->SetBranchAddress("r2tw", &r2tw, &b_r2tw);
   fChain->SetBranchAddress("h2mnrel", &h2mnrel, &b_h2mnrel);
   fChain->SetBranchAddress("h2mf", h2mf, &b_h2mf);
   fChain->SetBranchAddress("h2mt", h2mt, &b_h2mt);
   fChain->SetBranchAddress("h2mw", h2mw, &b_h2mw);
   Notify();
}

Bool_t MyLCTuple::Notify()
{
   // The Notify() function is called when a new file is opened. This
   // can be either for a new TTree in a TChain or when when a new TTree
   // is started when using PROOF. It is normally not necessary to make changes
   // to the generated code, but the routine can be extended by the
   // user if needed. The return value is currently not used.

   return kTRUE;
}

void MyLCTuple::Show(Long64_t entry)
{
// Print contents of entry.
// If entry is not specified, print current entry
   if (!fChain) return;
   fChain->Show(entry);
}
Int_t MyLCTuple::Cut(Long64_t entry)
{
// This function may be called from Loop.
// returns  1 if entry is accepted.
// returns -1 otherwise.
   return 1;
}
#endif // #ifdef MyLCTuple_cxx
