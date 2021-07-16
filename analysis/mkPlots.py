
import glob, ROOT, sys, math
from ROOT import TChain, TH1F, TH2F, TFile
from ROOT import TStyle,TGraph,TCanvas,TGaxis,TLegend,TGraphAsymmErrors,TPad,TLine
from ROOT import gROOT,gDirectory, gStyle
import numpy as np
import argparse

ROOT.gROOT.SetBatch(1)
ROOT.TH1.SetDefaultSumw2(1)
ROOT.gStyle.SetOptStat(0)


c1 = TCanvas("c1","c1",0,0,700,600)
axis_size = 0.035
title_offset=1.2
linewidth = 3
marker = 1.2
size = 12

signal_colors=[ROOT.kRed, ROOT.kBlue, ROOT.kOrange-3, ROOT.kMagenta, ROOT.kGreen+3, ROOT.kCyan, ROOT.kGreen, ROOT.kYellow, ROOT.kBlack]

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
        sig[i].Draw('histsames')
        sig[i].SetLineColor(signal_colors[i])
        sig[i].SetLineWidth(linewidth)

        if save.find('pos')>-1: leg.AddEntry(sig[i],legend[i]+', #sigma = %.2f'%(sig[i].GetRMS()) , 'l')
        else: leg.AddEntry(sig[i],legend[i], 'l')
        sig[i].GetXaxis().SetTitleSize(axis_size)
        sig[i].GetYaxis().SetTitleSize(axis_size)
        sig[i].GetYaxis().SetTitleOffset(title_offset)
        if max_val < sig[i].GetMaximum(): max_val =  sig[i].GetMaximum()
    sig[0].SetMaximum(max_val*1.5)

    #if save.find('hit_charge') > -1 : c1.SetLogy()
    leg.Draw()
    c1.Update()

    c1.cd()    
    c1.Modified()
    c1.Update()
    
    c1.Print(save+'.pdf')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--files",
        default="",
        help="Set the input configuration file",
    )
    parser.add_argument(
        "-l",
        "--legend",
        default ="",
        help='legend',
    )
    parser.add_argument(
        "-s",
        "--save",
        default ="",
        help='Name for saving the plot',
    )
    parser.add_argument(
        "-v",
        "--histograms",
        default="",
        help="Name of histogram to plot",
    )
    
    args = parser.parse_args()

    
    files = args.files.split(',')
    h = args.histograms
    legend= args.legend.split(',')
    save = args.save

    print(h)

    h_plt = []
    leg = []
    for f in files:
        tf = ROOT.TFile.Open('analysis_'+f+'.root')
        tmp = TH1F()
        tmp = tf.Get(h)
        tmp.SetDirectory(0)
        h_plt.append(tmp)
    plot_1var(h_plt,legend, h+'_'+save)
