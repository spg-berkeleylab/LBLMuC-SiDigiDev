
import glob, ROOT, sys, math
from ROOT import TChain, TH1F, TH2F, TFile
from ROOT import TStyle,TGraph,TCanvas,TGaxis,TLegend,TGraphAsymmErrors,TPad,TLine
from ROOT import gROOT,gDirectory, gStyle

ROOT.gROOT.SetBatch(1)
ROOT.TH1.SetDefaultSumw2(1)
ROOT.gStyle.SetOptStat(0)

files = glob.glob('analysis*.root')
print(files)

c1 = TCanvas("c1","c1",0,0,700,600)
axis_size = 0.035
title_offset=1.2
linewidth = 3
marker = 1.2
size = 12

signal_colors=[ROOT.kRed, ROOT.kBlue, ROOT.kOrange, ROOT.kMagenta, ROOT.kGreen+3, ROOT.kCyan, ROOT.kGreen, ROOT.kYellow, ROOT.kBlack]

def plot_1var(sig,legend, save):
# sig = array of TH1F histograms
# xaxis = x-axis title name 
# legend = array of leegnd titles
# save = name to save pdf
    c1.cd()

    if save.find('pos')>-1: leg = TLegend(0.2,0.65,0.4,0.85) 
    else: leg = TLegend(0.45,0.65,0.8,0.85)
    leg.SetNColumns(1)
    leg.SetTextFont(45)
    leg.SetTextSize(18)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)

    max_val = 0
    for i in range(0,len(sig)):
        sig[i].Draw('C sames')
        sig[i].SetLineColor(signal_colors[i])
        sig[i].SetLineWidth(linewidth)
        
        if save.find('pos')>-1: leg.AddEntry(sig[i],legend[i]+', #sigma = %.2f'%(sig[i].GetRMS()) , 'l')
        else: leg.AddEntry(sig[i],legend[i], 'l')
        sig[i].GetXaxis().SetTitleSize(axis_size)
        sig[i].GetYaxis().SetTitleSize(axis_size)
        sig[i].GetYaxis().SetTitleOffset(title_offset)

    leg.Draw()
    c1.Update()

    c1.cd()    
    c1.Modified()
    c1.Update()
    
    c1.Print(save+'.pdf')

histNames = ["clus_charge", "clus_posres_x", "clus_posres_y"]
histNames = sys.argv[1]
h = sys.argv[1]
if True:
    h_plt = []
    leg = []
    for f in files:
        if f.find('Poisson') > -1: continue
        if not(f.find('_4')> -1 or f.find('Threshold') >-1):continue
        l = f.split('.root')[0].split('analysis_')[1]
        if l.find('Threshold') > -1: leg.append('threshold smearing only')
        elif l.find('uniform') > -1: leg.append('uniform binning')
        elif l.find('variable') > -1 : leg.append('variable binning, bit = '+f.split('.root')[0].split('variableBin_')[1])
        elif l.find('binSplit') > -1: leg.append('Splitting bins, bit ='+f.split('.root')[0].split('binSplit_')[1])
        else: leg.append(l)
        tf = ROOT.TFile.Open(f)
        tmp = TH1F()
        tmp = tf.Get(h)
        tmp.SetDirectory(0)
        h_plt.append(tmp)
    plot_1var(h_plt,leg, h+'_chargeComp')
