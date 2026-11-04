import pandas as pd
import numpy as np
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
def determine_critical_temp(dataframe: pd.DataFrame, lo: float = 0.4, hi: float = 0.6):
    """
        Function for determining Tc, it takes the steep part of the curve, in the middle 
    
        kwargs:
        dataframe: the dataframe that is inserted into the funciton, expcected to be from MPMS3 (like fucntion thats used above)
        lo: the lower window, this can be tuned
        hi: the upper window, this can be tuned
        
        note:
        The default parameters are chosen 
        """
    df = dataframe.sort_values("Temperature (K)")
    temperature = df["Temperature (K)"].to_numpy()
    moment = df["DC Moment Fixed Ctr (emu)"].to_numpy()

    # m_sc = np.median(moment[:10])       
    # m_n = np.median(moment[-10:])       
    m_sc, m_n = np.percentile(moment, [5,95])
    moment_norm = (moment - m_n) / (m_sc - m_n)

    mask = (moment_norm < hi) & (moment_norm > lo)
    if mask.sum() < 3:
        return np.nan, np.full_like(temperature, np.nan), np.zeros_like(temperature)

    slope, intercept = np.polyfit(temperature[mask], moment_norm[mask], 1)
    fitted_line = slope * temperature + intercept

    t_crit = -intercept / slope if slope != 0 else np.nan
    return t_crit, fitted_line, np.zeros_like(temperature)

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

