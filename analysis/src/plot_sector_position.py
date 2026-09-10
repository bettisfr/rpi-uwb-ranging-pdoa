#!/usr/bin/env python3
"""Plot informative sector-classifier and oracle position results."""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'analysis'/'outputs'/'position_sector_lodo'
PLOTS=ROOT/'analysis'/'plots'

def main():
    PLOTS.mkdir(exist_ok=True)
    phase=pd.read_csv(OUT/'phase_metrics.csv').set_index('method').loc[['raw','sector','oracle']]
    pos=pd.read_csv(OUT/'position_metrics.csv').set_index('method').loc[['sector','oracle']]
    fig,axes=plt.subplots(1,2,figsize=(7.4,3.2),layout='constrained')
    axes[0].bar(['Raw PDoA','Sector\nclassifier','Bearing\noracle'],phase.mae_deg,color=['#616161','#1565c0','#8e24aa'])
    axes[0].set_ylabel('PDoA residual MAE [deg]'); axes[0].set_title('Phase correction'); axes[0].grid(axis='y',alpha=.25); axes[0].set_axisbelow(True)
    axes[1].bar(['Sector\nclassifier','Bearing\noracle'],pos.position_mae_m,color=['#1565c0','#8e24aa'])
    axes[1].set_ylabel('Position MAE [m]'); axes[1].set_title('Geometric position'); axes[1].grid(axis='y',alpha=.25); axes[1].set_axisbelow(True)
    fig.savefig(PLOTS/'position_sector_lodo.pdf',bbox_inches='tight')
    plt.close(fig)
    print(PLOTS/'position_sector_lodo.pdf')
if __name__=='__main__': main()
