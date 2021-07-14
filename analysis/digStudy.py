
import glob, ROOT, sys, math
from ROOT import TChain, TH1F, TH2F, TFile

ROOT.gROOT.SetBatch(1)
ROOT.TH1.SetDefaultSumw2(1)

path = '/global/homes/e/eressegu/LBLMuC-SiDigiDev/'
files = glob.glob(path+'*.root')
print(files)

for f in files:
    save_name = f.split('.root')[0].split('ntuple_tracker')[1]
    if save_name.find('defaultSettings') > -1: continue
    print(save_name)


    tf = ROOT.TFile.Open(f)
    t = tf.Get('MyLCTuple')

    electrons_per_GeV = 1. / 0.00362 * 1e6

    h=[]
    h.append(TH1F("clus_charge", "Cluster charge;charge (e^{-});a.u.", 50, 0.0, 20000.))
    h.append(TH1F("clus_charge_raw", "Cluster charge;charge (e^{-});a.u.", 1000, 0.0, 50000.))
    h.append(TH1F("hit_charge", "Hit Charge;charge (e^{-});a.u.", 100, 0.0, 10000.))
    h.append(TH1F("hit_charge_raw", "Hit Charge;charge (e^{-});a.u.", 500, 0.0, 10000.))
    h.append(TH1F("clus_posres_x", "Cluster position X (reco - true); #Delta x (#mu m);a.u.", 100, -50., 50.))
    h.append(TH1F("clus_posres_y", "Cluster position Y (reco - true); #Delta y (#mu m);a.u.", 100, -50., 50.))
    h.append(TH1F("clus_size_x", "Cluster size in X direction; Size (pixels); a.u.", 50, -0.5, 49.5))
    h.append(TH1F("clus_size_y", "Cluster size in Y direction; Size (pixels); a.u.", 50, -0.5, 49.5))
    h.append(TH2F("clus_size_y_theta", "Cluster size in Y direction vs #theta; #theta (deg); Size (pixels); a.u.", 140, 20.0, 160.0, 100, -0.5, 10.5))
    h.append(TH1F("hits_rem", "Percentage Removal (BIB); % of clusters removed; # of events", 100, 0.0, 100.0))

    #for hist in h:  hist.SetDirectory(0)

    for i in range (0, t.GetEntries()):
        t.GetEntry(i)

        for clus in range (0,t.ntrh):
            truth = t.h2mt[clus]
            clusCharge = t.thedp[clus] * electrons_per_GeV

            ROOT.gDirectory.Get("clus_charge").Fill(clusCharge)
            ROOT.gDirectory.Get("clus_charge_raw").Fill(clusCharge)
            ROOT.gDirectory.Get("clus_posres_x").Fill((t.thpox[clus] - t.stpox[truth]) * 1e3)
            ROOT.gDirectory.Get("clus_posres_y").Fill((t.thpoy[clus] - t.stpoy[truth]) * 1e3)

            minX = sys.maxsize
            maxX = -1*sys.maxsize
            minY = sys.maxsize
            maxY = -1*sys.maxsize

            for hit in range(t.thcidx[clus], t.thcidx[clus] + t.thclen[clus]):
                ROOT.gDirectory.Get("hit_charge").Fill(t.tcedp[hit])
                ROOT.gDirectory.Get("hit_charge_raw").Fill(t.tcedp[hit])
                minX = min(minX, t.tcrp0[hit])
                maxX = max(maxX, t.tcrp0[hit])
                minY = min(minY, t.tcrp1[hit])
                maxY = max(maxY, t.tcrp1[hit])

            sizeX = abs(maxX - minX) + 1
            sizeY = abs(maxY - minY) + 1
            theta = math.atan2(t.stpoy[truth], t.stpoz[truth]) / math.pi * 180.

            if theta < 0: theta += 180

            ROOT.gDirectory.Get("clus_size_x").Fill(sizeX)
            ROOT.gDirectory.Get("clus_size_y").Fill(sizeY)
            ROOT.gDirectory.Get("clus_size_y_theta").Fill(theta, sizeY)

    # study for threshold smear only, no charge digi
    if save_name.find('ThresholdSmear') > -1:
        h_charge = ROOT.gDirectory.Get("hit_charge_raw")
        lastBin = h_charge.GetSize()-1
        totalIntegral = h_charge.Integral(0, 15000)
        lastBin = h_charge.FindBin(15000)
        nBits = [3,4,5,6,8,10]
        
        for j in range(0, len(nBits)):
            nBins = pow(2, nBits[j])
            
            bins=[0]
            cnt=0
        
            for iBin in range(0, lastBin):
                tmp = h_charge.Integral(bins[cnt], iBin)            
                if tmp >= totalIntegral/nBins:
                    cnt+=1
                    bins.append(iBin)            
            print('bit', nBits[j])
            tmp1=[]
            tmp2=[]
            for k in range(1,len(bins)):
                tmp1.append(h_charge.Integral(bins[k-1],bins[k]))
                tmp2.append(h_charge.GetBinCenter(bins[k]))
            #print(tmp1)
            print(tmp2)

            
    outName = 'analysis'+save_name+'.root'
    outputFile = TFile(outName, "RECREATE");
    for hist in h: hist.Write()
    outputFile.Close()
