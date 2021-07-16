import glob, ROOT, sys, math
from ROOT import TChain, TH1F, TH2F, TFile
import array

ROOT.gROOT.SetBatch(1)
ROOT.TH1.SetDefaultSumw2(1)

path = '/global/homes/e/eressegu/LBLMuC-SiDigiDev/'
files = glob.glob(path+'*.root')

for f in files:
    save_name = f.split('.root')[0].split('ntuple_tracker')[1]
    if save_name.find('defaultSettings') > -1: continue
    if f.find('ntuple_tracker_BIB.root') > -1: continue
    
    print(save_name)
    
    nbins = []
    if save_name.find('uniform') > -1:
        if save_name.find('4') > -1: nbins = array.array('d',[0,500,1467,2433,3400,4367,5333,6300,7267,8233,9200,10167,11133,12100,13067,14033,15000])
    if save_name.find('variableBin') > -1:
        if save_name.find('3') > -1: nbins=array.array('d',[0,500, 786, 1100, 1451, 1854, 2390, 3326, 31973, 50000])
        if save_name.find('4') > -1: nbins=array.array('d',[0, 500, 639, 769, 910, 1057, 1213, 1379, 1559, 1743, 1945, 2193, 2484, 2849, 3427, 4675, 29756, 50000])
        if save_name.find('6') > -1: nbins=array.array('d',[0,500, 542, 572, 601, 629, 661, 692, 721, 750, 779, 812, 842, 877, 913, 946, 981, 1016, 1051, 1087, 1121, 1161, 1196, 1237, 1275, 1313, 1350, 1391, 1431, 1468, 1514, 1560, 1606, 1646, 1687, 1733, 1777, 1821, 1872, 1920, 1976, 2036, 2091, 2145, 2213, 2272, 2337, 2411, 2488, 2573, 2651, 2739, 2834, 2938, 3053, 3194, 3356, 3532, 3764, 4034, 4379, 4907, 5698, 6957, 9636, 50000])
        if save_name.find('8') > -1: nbins=array.array('d',[0, 500, 511, 523, 533, 542, 550, 556, 564, 570, 577, 585, 592, 598, 603, 610, 617, 624, 630, 638, 646, 654, 661, 668, 676, 684, 691, 699, 705, 712, 719, 724, 731, 738, 745, 752, 760, 767, 772, 780, 787, 795, 802, 810, 818, 826, 832, 839, 847, 856, 865, 874, 881, 889, 898, 906, 916, 924, 930, 938, 945, 955, 965, 971, 978, 986, 995, 1004, 1012, 1019, 1027, 1036, 1044, 1053, 1062, 1071, 1079, 1088, 1096, 1104, 1112, 1121, 1131, 1139, 1149, 1158, 1168, 1175, 1184, 1193, 1203, 1211, 1221, 1233, 1241, 1249, 1259, 1268, 1277, 1286, 1294, 1303, 1313, 1321, 1330, 1338, 1348, 1357, 1368, 1378, 1387, 1395, 1406, 1417, 1426, 1434, 1445, 1452, 1460, 1470, 1480, 1492, 1503, 1514, 1525, 1536, 1550, 1560, 1570, 1580, 1592, 1604, 1614, 1623, 1634, 1644, 1653, 1662, 1673, 1684, 1695, 1707, 1717, 1727, 1737, 1747, 1759, 1769, 1780, 1790, 1800, 1812, 1823, 1835, 1846, 1860, 1873, 1885, 1897, 1907, 1918, 1931, 1943, 1958, 1971, 1987, 2000, 2014, 2026, 2041, 2056, 2068, 2080, 2095, 2108, 2119, 2131, 2147, 2162, 2180, 2195, 2213, 2224, 2238, 2256, 2269, 2284, 2300, 2314, 2332, 2351, 2366, 2383, 2401, 2421, 2440, 2458, 2475, 2496, 2519, 2538, 2559, 2581, 2601, 2618, 2636, 2658, 2681, 2703, 2722, 2742, 2767, 2791, 2811, 2836, 2857, 2884, 2913, 2938, 2967, 2995, 3023, 3052, 3086, 3119, 3153, 3188, 3221, 3270, 3304, 3342, 3390, 3428, 3473, 3515, 3556, 3611, 3691, 3742, 3801, 3857, 3928, 3999, 4069, 4141, 4220, 4325, 4417, 4518, 4655, 4789, 4965, 5141, 5359, 5548, 5770, 6017, 6311, 6584, 7024, 7492, 8060, 8740, 9738, 11450, 14878, 23973,50000])

    tf = ROOT.TFile.Open(f)
    t = tf.Get('MyLCTuple')

    electrons_per_GeV = 1. / 0.00362 * 1e6

    h=[]
    h.append(TH1F("clus_charge", "Cluster charge;charge (e^{-});a.u.", 50, 0.0, 20000.))

    if len(nbins) > 0 and save_name.find('var') >-1:        
        hitcharge_prebin = TH1F("hit_charge_prebin", "Hit Charge;charge (e^{-});a.u.", 50000, 0.0, 50000.)
        hitcharge = hitcharge_prebin.Rebin(len(nbins)-1, "hit_charge", nbins)
        h.append(hitcharge)
    elif len(nbins) > 0 and save_name.find('uniform') >-1:
        hitcharge_prebin = TH1F("hit_charge_prebin", "Hit Charge;charge (e^{-});a.u.", 15000, 0.0, 15000.)
        hitcharge = hitcharge_prebin.Rebin(len(nbins)-1, "hit_charge", nbins)
        h.append(hitcharge)
    else:
        hitcharge = TH1F("hit_charge", "Hit Charge;charge (e^{-});a.u.", 100, 0.0, 10000.)
        h.append(hitcharge)
    
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
            ROOT.gDirectory.Get("clus_posres_x").Fill((t.thpox[clus] - t.stpox[truth]) * 1e3)
            ROOT.gDirectory.Get("clus_posres_y").Fill((t.thpoy[clus] - t.stpoy[truth]) * 1e3)

            minX = sys.maxsize
            maxX = -1*sys.maxsize
            minY = sys.maxsize
            maxY = -1*sys.maxsize

            for hit in range(t.thcidx[clus], t.thcidx[clus] + t.thclen[clus]):
                ROOT.gDirectory.Get("hit_charge").Fill(t.tcedp[hit])
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
            
    outName = 'analysis'+save_name+'.root'
    outputFile = TFile(outName, "RECREATE");
    for hist in h: hist.Write()
    outputFile.Close()
