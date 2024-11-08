import pandas as pd
import matplotlib.pyplot as plt
from openap.traj.gen import Generator
import numpy as np


def plot_traj(flights: pd.DataFrame | list[pd.DataFrame], lw=2):
    if isinstance(flights, pd.DataFrame):
        flights = [flights]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 4))

    for f in flights:
        print(f)
        ax1.plot(f.t, f.altitude, lw=lw)
        ax2.plot(f.t, f.groundspeed, lw=lw)
        ax3.plot(f.t, f.vertical_rate, lw=lw)
        ax4.plot(f.t, f.s / 1000, lw=lw)

        ax1.set_ylabel("altitude (ft)")
        ax2.set_ylabel("groundspeed (kt)")
        ax3.set_ylabel("vertical rate (ft/min)")
        ax4.set_ylabel("distance flown (km)")

        ax1.set_ylim(-1000, 40_000)
        ax2.set_ylim(0, 600)
        ax3.set_ylim(-3000, 3000)

    for ax in (ax1, ax2, ax3, ax4):
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.yaxis.set_label_coords(-0.1, 1.05)
        ax.yaxis.label.set_rotation(0)
        ax.yaxis.label.set_ha("left")
        ax.grid()

        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    fgen = Generator(ac="a320")
    flight = fgen.complete(dt=10)
    #display(flight)
    print(flight)
    plot_traj(flight)
    
