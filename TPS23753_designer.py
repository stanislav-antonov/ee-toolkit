# https://www.ti.com/lit/an/slva305c/slva305c.pdf?ts=1777622956261

import numpy as np
import matplotlib.pyplot as plt

class Designer:
    def __init__(self):
        # ---- Output ----
        self.V_OUT_NOM = 3.3
        self.V_OUT_MIN = 3.15
        self.V_OUT_RIPPLE = 0.05 # 50mV ripple
        self.P_OUT_MAX = 3.0
        self.P_OUT_MIN = 0
        self.I_OUT_MAX = 0.91
        self.I_OUT_MIN = 0

        # ---- Input ----
        self.V_PSE_MAX = 57
        self.V_PSE_MIN = 44
        self.I_PSE_MAX = 0.35
        self.I_PSE_MIN = 0
        self.V_ADP_NOM = 24
        self.V_ADP_MIN = 21.6

        # ---- Switching frequency ----
        self.FREQ_NOM = 250e3
        
        # ---- Efficiency target ----
        self.N_FB = 0.78

        # ---- Duty cycle ----
        self.D_MAX_DESIGN = 0.6
        

        # ---- Frequency setting resistor ----
        self.R_FRS = 15000000 / self.FREQ_NOM

        # ---- Blanking interval setting resistor ----
        self.BI = 2 # 2% 
        self.R_BLNK= (self.BI/self.FREQ_NOM) * 10e4


        self.j = 1j

        # ---- Maximum adapter input current ----
        self.I_ADP_MAX = self.P_OUT_MAX / (self.V_ADP_MIN * self.N_FB)

        # ---- Minimum converter input voltage ----
        self.V_FLYBACK_MIN = self.V_ADP_MIN - 0.7

        #
        # ==== Flyback transformer ====
        #
        
        # ---- Primary voltage drops ----
        self.R_DS_ON = 0.5
        self.R_CS = 1.2
        self.V_DROP_PRIMARY = 2 * self.I_ADP_MAX * (self.R_DS_ON + self.R_CS) 

        # ---- Secondary and bias voltage drops ----
        self.I_SECONDARY = (self.V_FLYBACK_MIN / self.V_OUT_MIN) * 2 * self.I_ADP_MAX
        self.V_DROP_SECONDARY = 0.4
        
        self.V_DVC = 0.5
        self.I_VC = 0.005
        self.R_VC = 50
        self.V_DROP_BIAS = self.V_DVC + self.I_VC * self.R_VC

        self.V_BIAS_NOM = 12

        # ---- Required transformer turns ratios ----
        self.N_P_S = (self.D_MAX_DESIGN / (1 - self.D_MAX_DESIGN)) * ((self.V_FLYBACK_MIN - self.V_DROP_PRIMARY) / (self.V_OUT_NOM + self.V_DROP_SECONDARY))
        self.N_P_B = (self.D_MAX_DESIGN / (1 - self.D_MAX_DESIGN)) * ((self.V_FLYBACK_MIN - self.V_DROP_PRIMARY) / (self.V_DROP_BIAS + self.V_BIAS_NOM))

        # ---- Target peak primary current ----
        self.I_PEAK = 4/3 * ((self.I_OUT_MAX / self.N_P_S) * (1 / (1 - self.D_MAX_DESIGN)))

        # ---- Primary inductance  ----
        self.L_PRIM = (self.D_MAX_DESIGN / self.FREQ_NOM) * ((self.V_FLYBACK_MIN - self.V_DROP_PRIMARY) / (0.5 * self.I_PEAK))

        # ---- Select the transformer
        self.L_P = 310e-6    # Primary inductance
        self.N_PS = 1 / 0.19 # Turns ratio primary to secondary
        self.N_PB = 1 / 0.7  # Turns ratio primary to bias
        self.L_LKG = 4.75e-6 # Leakage inductance

        # ---- Actual duty cycle at vlotage using selected transformer ----
        self.D_MAX_ACTUAL = ((self.V_OUT_NOM + self.V_DROP_SECONDARY) * self.N_PS) / ((self.V_FLYBACK_MIN - self.V_DROP_PRIMARY) + ((self.V_OUT_NOM + self.V_DROP_SECONDARY) * self.N_PS))

        # ---- Primary currents ----
        self.I_DCFB_MAX = self.P_OUT_MAX / (self.V_FLYBACK_MIN * self.N_FB)
        self.I_PRI_STEP = self.I_DCFB_MAX / self.D_MAX_ACTUAL
        self.DELTA_I_L_PRIMARY = ((self.V_FLYBACK_MIN - self.V_DROP_PRIMARY) / self.L_P) * (self.D_MAX_ACTUAL / self.FREQ_NOM)
        self.I_PRIMARY_PEAK = self.I_PRI_STEP + (self.DELTA_I_L_PRIMARY / 2)

        #
        # ==== Power train ====
        #

        # ---- Current sense resistor ----
        self.R_CS_MIN = 0.55 / self.I_PRIMARY_PEAK
        self.R_CS = 1.24 

        # ---- Snubber ----
        self.C_WDG = 200e-12 
        self.V_SPIKE = self.I_PRIMARY_PEAK * np.sqrt(self.L_LKG / self.C_WDG) 
        self.V_SPIKE_NEW  = 25
        self.C_SN_MIN = ((self.V_SPIKE / self.V_SPIKE_NEW)**2) * self.C_WDG # snubber capacitance
        self.C_SN = 4.7e-9 # selected snubber capacitance
        self.R_SN = 200 / (self.FREQ_NOM * self.C_SN)                   # snubber resistor

        # ---- Input filter ----
        self.V_IN_RIPPLE = 1
        self.C_IN_MIN = ((self.I_PRI_STEP - self.I_DCFB_MAX) * self.D_MAX_ACTUAL) / (self.FREQ_NOM * self.V_IN_RIPPLE)
        self.C_IN2 = 1e-6
        self.ESR_C_IN2 = 10e-3

        self.DELTA_V_C_IN2 = (((self.I_PRI_STEP - self.I_DCFB_MAX) * self.D_MAX_ACTUAL) / (self.FREQ_NOM * self.C_IN2)) + (self.I_PRI_STEP * self.ESR_C_IN2)

        self.C_IN1 = 22e-6
        self.ESR_C_IN1 = 1.3
        self.DELTA_I_C_IN1 = 0.13

        self.DELTA_V_C_IN1 = self.DELTA_I_C_IN1 * self.ESR_C_IN1

        self.L_IN = ((self.DELTA_V_C_IN1 + self.DELTA_V_C_IN2) / (self.I_PRI_STEP - self.I_DCFB_MAX - self.DELTA_I_C_IN1)) * (self.D_MAX_ACTUAL / self.FREQ_NOM)

        # ---- Secondary currents -----
        self.I_SEC_STEP = self.I_OUT_MAX / (1 - self.D_MAX_ACTUAL)
        self.DELTA_I_L_SECONDARY = 2 * ((self.N_PS * self.I_PRIMARY_PEAK) - self.I_SEC_STEP) / self.FREQ_NOM
        self.I_SECONDARY_PEAK = self.N_PS * self.I_PRIMARY_PEAK

        # ---- Output filter ----
        self.C_OUT_MIN = ((self.I_SEC_STEP - self.I_OUT_MAX) * (1 - self.D_MAX_ACTUAL)) / (self.FREQ_NOM * self.V_OUT_RIPPLE)
        self.C_OUT2 = 2 * 47e-6
        self.C_OUT2_ESR = 2e-3
        self.DELTA_V_C_OUT2 = (((self.I_SEC_STEP - self.I_OUT_MAX) * (1 - self.D_MAX_ACTUAL)) / (self.FREQ_NOM * self.C_OUT2)) + (self.I_SEC_STEP - self.I_OUT_MAX) * self.C_OUT2_ESR
        self.C_OUT1 = 47e-6
        self.C_OUT1_ESR = 1.25e-3

        #
        # ==== Feedback loop ====
        #

        # ---- Shunt regulator ----
        self.V_REF = 1.24

        # ---- Output voltage setpoint resistors ----
        # Choose RFBU based on expected integrator midband gain and zero location. Estimate the integrator zero
        # location to be approximately 1 kHz, integrator zero capacitors are from 1 nF–10 nF, and low integrator
        # gain.
        self.F_IZ = 1e3
        self.C_IZ_E = 4.7e-9
        self.R_IZ_E = 1 / (2 * np.pi * self.F_IZ * self.C_IZ_E)
        
        self.R_FBU_E = self.R_IZ_E
        self.R_FBU = 41.2e3

        self.R_FBL_E = (self.V_REF * self.R_FBU) / (self.V_OUT_NOM - self.V_REF)
        self.R_FBL = 24.3e3

        # ---- Opto-Isolator ----
        self.CTR_2mA = 0.85 # from datasheet, at 2mA LED current
        self.V_LED_OPTO = 1.1
        self.I_LED_OPTO = 2e-3

        # ---- Opto-Isolator LED Bias Current ----
        self.V_LED_OPTO_K = self.V_LED_OPTO + 0.15
        self.R_OB_E = (self.V_OUT_NOM - self.V_LED_OPTO - self.V_LED_OPTO_K) / self.I_LED_OPTO
        self.R_OB = 402

        # ---- Opto-Isolator transistor bias current ----
        self.V_ZDC_MAX = 1.7
        self.V_CS_MAX = 0.55
        self.V_B = 5
        self.V_CTL_MAX = self.V_ZDC_MAX + 2 * self.V_CS_MAX
        self.V_CTL_MOM = (self.V_ZDC_MAX + self.V_CTL_MAX) / 2
        
        self.R_CTL_E = (self.V_B - self.V_ZDC_MAX) / (self.I_LED_OPTO * self.CTR_2mA)
        self.R_CTL = 1.5e3
        
        #
        # ==== Frequency characteristics ====
        #

        # ---- Modulator power stage (MPS) gain and right-half-plane zero (RHPZ) ----
        self.K_MPS = ((1 - self.D_MAX_ACTUAL) * self.N_PS) / self.R_CS
        self.R_LOAD = (self.V_OUT_NOM ** 2) / self.P_OUT_MAX
        self.RHPZ = self.R_LOAD * ( ((self.N_PS * (1 - self.D_MAX_ACTUAL)) ** 2) / (2 * np.pi * self.D_MAX_ACTUAL * self.L_P) )

        #
        # ==== Loop compensation ====
        #

        # ---- Crossover frequency ----
        self.F0 = 5.5e3
        self.w0 = 2 * np.pi * self.F0

        # ---- Overall transfer function ----
        self.K_CTL = 2

        # ---- Compensate opto-isolator or inner loop ----
        self.G_MO = 0.75 # initial estimate at F0
        self.C_CTL_E = self.C_ctl(self.w0, self.G_MO)
        self.C_CTL = 6.8e-9

        # ---- Inner loop control zero resistor ----
        self.R_ZCTL_E = self.R_CTL / 10
        self.R_ZCTL = 160

        # ----- Outer loop compensation ----
        self.G_MO = np.abs(self.gmo(self.w0))
        self.R_IZ = self.R_FBU * ((1 / self.G_MO) - 1)

        self.C_IZ = 5 / (2 * np.pi * self.R_IZ * self.F0)
        self.C_IP = 1 / (20 * np.pi * self.R_IZ * self.F0)
    
    def fmt(self, v, unit=""):
        v_abs = abs(v)

        if v_abs >= 1e9:
            return f"{v/1e9:.3g} G{unit}"
        elif v_abs >= 1e6:
            return f"{v/1e6:.3g} M{unit}"
        elif v_abs >= 1e3:
            return f"{v/1e3:.3g} k{unit}"
        elif v_abs >= 1:
            return f"{v:.3g} {unit}"
        elif v_abs >= 1e-3:
            return f"{v*1e3:.3g} m{unit}"
        elif v_abs >= 1e-6:
            return f"{v*1e6:.3g} µ{unit}"
        elif v_abs >= 1e-9:
            return f"{v*1e9:.3g} n{unit}"
        else:
            return f"{v:.3g} {unit}"
    
    def C_ctl(self, w, gmo_target=0.75):
        x = (self.R_CTL / self.R_OB) * (self.CTR_2mA / self.K_CTL) * (abs(self.mpf(w)) / gmo_target)
        
        if x >= 1:
            raise ValueError(
                f"No valid solution: loop gain too high at F0 → x = {x:.2f} ≥ 1\n"
                f"→ Reduce crossover frequency or loop gain"
            )

        value = np.sqrt(1 - x**2) / ( w * self.R_CTL)
        
        if value > 47e-9:
            raise ValueError(f"Rctl capacitor value {self.fmt(value, 'F')} is too high")
        
        return value

    # =========================
    # MPS current
    # ========================
    def I_mps(self, w):
        w_rhpz = 2 * np.pi * self.RHPZ
        return self.K_MPS * (1 - self.j*w / w_rhpz)

    # ========================
    # Output impedance
    # ========================
    def Z_out(self, w):
        j = self.j

        Zc1 = 1 / (j*w * self.C_OUT1) + self.C_OUT1_ESR
        Zc2 = 1 / (j*w * self.C_OUT2) + self.C_OUT2_ESR
        Zout = 1 / (1/Zc1 + 1/Zc2 + 1/self.R_LOAD)

        return Zout
    
    # =========================
    # MPF transfer function
    # =========================
    def mpf(self, w):
        return self.I_mps(w) * self.Z_out(w)

    def opto_simplified(self, w):
        return (self.R_CTL / self.R_OB) * self.CTR_2mA * (1 / (1 + self.j * w * self.C_CTL * self.R_CTL)) * (1 / self.K_CTL)

    def gmo_simplified(self, w):
        return self.mpf(w) * self.opto_simplified(w)
    
    def opto(self, w):
        return (self.R_CTL / self.R_OB) * self.CTR_2mA * ((1 + self.j * w * self.R_ZCTL * self.C_CTL) / (1 + self.j * w * self.C_CTL * (self.R_CTL + self.R_ZCTL))) * (1 / self.K_CTL)
    
    def gmo(self, w):
        return self.mpf(w) * self.opto(w)
    
    def integrator_simplified(self, w):
        return 1 / (self.j * w * self.R_FBU * self.C_IZ)
    
    def integrator(self, w):
        return (self.R_IZ / self.R_FBU) * ((1 + (1 / (self.j * w * self.R_IZ * self.C_IZ))) / (1 + self.j * w * self.R_IZ * self.C_IP))

    def pretty(self):
        print("\n=== DESIGN SUMMARY ===\n")

        print("---- Output ----")
        print(f"Vout        : {self.fmt(self.V_OUT_NOM, 'V')}")
        print(f"Iout max    : {self.fmt(self.I_OUT_MAX, 'A')}")
        print(f"Pout max    : {self.fmt(self.P_OUT_MAX, 'W')}")
        print(f"Ripple      : {self.fmt(self.V_OUT_RIPPLE, 'V')}")

        print("\n---- Input ----")
        print(f"PSE range      : {self.fmt(self.V_PSE_MIN, 'V')} - {self.fmt(self.V_PSE_MAX, 'V')}")
        print(f"Adapter        : {self.fmt(self.V_ADP_NOM, 'V')} (min {self.fmt(self.V_ADP_MIN, 'V')})")

        print("\n---- Transformer ----")
        print(f"L_primary(est.): {self.fmt(self.L_PRIM, 'H')}")
        print(f"L_primary      : {self.fmt(self.L_P, 'H')}")
        print(f"Np:Ns(est.)    : {self.N_P_S:.2f}")
        print(f"Np:Ns          : {self.N_PS:.2f}")
        print(f"Np:Nbias(est.) : {self.N_P_B:.2f}")
        print(f"Np:Nbias       : {self.N_PB:.2f}")
        print(f"Ipk primary    : {self.fmt(self.I_PRIMARY_PEAK, 'A')}")

        print("\n---- Primary Side ----")
        print(f"Rcs(est.)      : {self.fmt(self.R_CS_MIN, 'Ω')}")
        print(f"Rcs            : {self.fmt(self.R_CS, 'Ω')}")
        print(f"Fswitch        : {self.fmt(self.FREQ_NOM, 'Hz')}")
        print(f"Duty max       : {self.D_MAX_ACTUAL*100:.1f} %")

        print("\n---- Snubber ----")
        print(f"Csn(est.)      : {self.fmt(self.C_SN_MIN, 'F')}")
        print(f"Csn            : {self.fmt(self.C_SN, 'F')}")
        print(f"Rsn            : {self.fmt(self.R_SN, 'Ω')}")

        print("\n---- Input Filter ----")
        print(f"Cin min        : {self.fmt(self.C_IN_MIN, 'F')}")
        print(f"Cin1           : {self.fmt(self.C_IN1, 'F')}")
        print(f"Cin2           : {self.fmt(self.C_IN2, 'F')}")
        print(f"Lin            : {self.fmt(self.L_IN, 'H')}")

        print("\n---- Output Filter ----")
        print(f"Cout min       : {self.fmt(self.C_OUT_MIN, 'F')}")
        print(f"Cout1          : {self.fmt(self.C_OUT1, 'F')}")
        print(f"Cout2          : {self.fmt(self.C_OUT2, 'F')}")

        print("\n======================\n")

        print("\n=== FEEDBACK CONTROL ===\n")

        print("---- Shunt regulator ----")
        print(f"Vref            : {self.fmt(self.V_REF, 'V')}")
        
        print("\n---- Voltage setpoint resistors ----")
        print(f"R_FBU(est.)     : {self.fmt(self.R_FBU_E, 'Ω')}")
        print(f"R_FBU           : {self.fmt(self.R_FBU, 'Ω')}")
        
        print(f"R_FBL(est.)     : {self.fmt(self.R_FBL_E, 'Ω')}")
        print(f"R_FBL           : {self.fmt(self.R_FBL, 'Ω')}")

        print("\n---- Opto-Isolator ----")
        print(f"R_OB(est.)      : {self.fmt(self.R_OB_E, 'Ω')}")
        print(f"R_OB            : {self.fmt(self.R_OB, 'Ω')}")
        
        print(f"R_CTL(est.)     : {self.fmt(self.R_CTL_E, 'Ω')}")
        print(f"R_CTL           : {self.fmt(self.R_CTL, 'Ω')}")
        
        print(f"C_CTL(est.)     : {self.fmt(self.C_CTL_E, 'F')}")
        print(f"C_CTL           : {self.fmt(self.C_CTL, 'F')}")
        
        print(f"R_ZCTL(est.)    : {self.fmt(self.R_ZCTL_E, 'Ω')}")
        print(f"R_ZCTL          : {self.fmt(self.R_ZCTL, 'Ω')}")

        print("\n---- Loop compensation ----")
        print(f"F0              : {self.fmt(self.F0, 'Hz')}")
        print(f"G_MO            : {self.G_MO:.2f}")
        print(f"R_IZ            : {self.fmt(self.R_IZ, 'Ω')}")

        


    def plot_mpf(self, fmin=1e3, fmax=1e6, points=1000):
        freqs = np.logspace(np.log10(fmin), np.log10(fmax), points)
        w = 2 * np.pi * freqs

        mpf_vals = np.array([self.mpf(wi) for wi in w])

        mag = 20 * np.log10(np.abs(mpf_vals))
        phase = np.angle(mpf_vals, deg=True)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

        # ---- Magnitude ----
        ax1.semilogx(freqs, mag)
        ax1.set_ylabel("MPF Gain (dB)")
        ax1.grid(True)

        # ---- Phase ----
        ax2.semilogx(freqs, phase)
        ax2.set_ylabel("Phase (deg)")
        ax2.set_xlabel("Frequency (Hz)")
        ax2.grid(True)

        plt.suptitle("Modulator Power Stage (MPF) Bode Plot")
        plt.show()

    
designer = Designer()
designer.pretty()
designer.plot_mpf()
