import pandas as pd
import numpy as np
from scipy.stats import theilslopes

#=============================
#  Read Squid data function  #
#=============================
def read_squid_file(filepath: str) -> pd.DataFrame:
    """
    A function for reading in Squid Files 

    kwargs:
    filepath -- the string filepath for the dataframe
    """
    with open(filepath, encoding="utf-8", errors="replace") as f:
        skip = next(
            i for i, line in enumerate(f, start=1) if line.strip() == "[Data]"
        )

    df = pd.read_csv(
        filepath,
        skiprows=skip,
        na_values=["No data", "NaN", "", " ", "  ", "m"],
    )

    df = df.drop(
        columns=[
            "Comment",
            "Averaging Time (sec)",
            "Frequency (Hz)",
            "Peak Amplitude (mm)",
            "Lockin Signal' (V)",
            "Lockin Signal (V)",
            "Range",
            "M. Quad. Signal (emu)",
            "AC Moment (emu)",
            "AC M. Std Err. (emu)",
            "AC Phase (deg)",
            "AC Phase Std. Err. (deg)",
            "Map 01",
            "Map 02",
            "Map 03",
            "Map 04",
            "Map 05",
            "Map 06",
            "Map 07",
            "Map 08",
            "Map 09",
            "Map 10",
            "Map 11",
            "Map 12",
            "Map 13",
            "Map 14",
            "Map 15",
            "Map 16",
        ]
    )
    return df

#=============================
#     Find T_c Function      #
#=============================

def determine_critical_temp(dataframe: pd.DataFrame,
                            normal_pct: float = 70,
                            trans_lo: float = 0.55,
                            trans_hi: float = 0.95,
                            min_points: int = 3):
    """
        Function for determining Tc from the intersection of two robust (Theil-Sen) lines:
        one fitted to the normal state, one fitted to the steep part of the transition.

        kwargs:
        dataframe: single-run dataframe, expected to be from MPMS3
        normal_pct: temperature percentile above which data is treated as normal state
        trans_lo: lower normalised moment bound of the transition window (0 = SC, 1 = normal)
        trans_hi: upper normalised moment bound of the transition window
        min_points: minimum points required in each region

        returns:
        t_crit: critical temperature (np.nan if the fit fails)
        fit: dict of the two lines (slope, intercept) and the masks, for plotting later
        """
    df = dataframe.sort_values("Temperature (K)").drop_duplicates(subset="Temperature (K)")
    temperature = df["Temperature (K)"].to_numpy(dtype=float)
    moment = df["DC Moment Fixed Ctr (emu)"].to_numpy(dtype=float)

    # Normalise moment: 0 = superconducting, 1 = normal
    m_sc, m_n = np.percentile(moment, [5, 95])
    moment_norm = (moment - m_sc) / (m_n - m_sc)

    # Normal state selected on temperature only; transition on the moment window
    t_limit = np.percentile(temperature, normal_pct)
    normal_mask = temperature >= t_limit
    trans_mask = (moment_norm >= trans_lo) & (moment_norm < trans_hi) & ~normal_mask

    fit = {"normal_mask": normal_mask, "trans_mask": trans_mask,
           "normal": (np.nan, np.nan), "trans": (np.nan, np.nan)}

    if normal_mask.sum() < min_points or trans_mask.sum() < min_points:
        return np.nan, fit

    n_slope, n_int, *_ = theilslopes(moment[normal_mask], temperature[normal_mask])
    t_slope, t_int, *_ = theilslopes(moment[trans_mask], temperature[trans_mask])
    fit["normal"], fit["trans"] = (n_slope, n_int), (t_slope, t_int)

    # Tc = intersection of the two lines
    t_crit = (t_int - n_int) / (n_slope - t_slope) if n_slope != t_slope else np.nan
    return t_crit, fit

#=============================
#      Find H_c Function     #
#=============================
    """
        Function for determining H_c, it determines a noise floor, determined by the user,
        when the material reaches the critical field, the magnetic moment increases/decreases
        therefore we can track the point where this happens, and note it as critical field 
    
        kwargs:
        dataframe: the dataframe that is inserted into the funciton, expcected to be from MPMS3 (like fucntion thats used above)
        threshold: where we decide the field is definitely critical, and not just a noise spike
        
        note:
        The default parameters are chosen 
        """
def determine_critical_field(dataframe: pd.DataFrame, threshold: float = 0.005):
    df = dataframe.sort_values("Magnetic Field (Oe)")
    field = df["Magnetic Field (Oe)"].to_numpy()
    moment = df["DC Moment Fixed Ctr (emu)"].to_numpy()
    cutoff  = threshold * np.max(np.abs(moment))

    field_pos = field[moment < -cutoff].max()   
    field_neg = field[moment > cutoff].min()  
    f_crit = (field_pos - field_neg) / 2
    return f_crit, field_pos, field_neg
