import os
import scipy.stats as ss
from bisect import bisect_left


def VD_A(treatment, control):
    """
    Computes Vargha and Delaney A index
    A. Vargha and H. D. Delaney.
    A critique and improvement of the CL common language
    effect size statistics of McGraw and Wong.
    Journal of Educational and Behavioral Statistics, 25(2):101-132, 2000
    The formula to compute A has been transformed to minimize accuracy errors
    See: http://mtorchiano.wordpress.com/2014/05/19/effect-size-of-r-precision/
    :param treatment: a numeric list
    :param control: another numeric list
    :returns the value estimate and the magnitude and power and number of runs to have power>0.8
    """
    m = len(treatment)
    n = len(control)

    # if m != n:
    #     raise ValueError("Data must have the same length")

    r = ss.rankdata(treatment + control)
    r1 = sum(r[0:m])

    # Compute the measure
    # A = (r1/m - (m+1)/2)/n # formula (14) in Vargha and Delaney, 2000
    A = (2 * r1 - m * (m + 1)) / (2 * n * m)  # equivalent formula to avoid accuracy errors

    levels = [0.147, 0.33, 0.474]  # effect sizes from Hess and Kromrey, 2004
    magnitude = ["negligible", "small", "medium", "large"]
    scaled_A = (A - 0.5) * 2

    magnitude = magnitude[bisect_left(levels, abs(scaled_A))]
    estimate = A

    return estimate, magnitude#, power, nruns



def _log_raw_statistics(treatment, treatment_name, control, control_name):
    # Compute p : In statistics, the Mann–Whitney U test (also called the Mann–Whitney–Wilcoxon (MWW),
    # Wilcoxon rank-sum test, or Wilcoxon–Mann–Whitney test) is a nonparametric test of the null hypothesis that,
    # for randomly selected values X and Y from two populations, the probability of X being greater than Y is
    # equal to the probability of Y being greater than X.

    statistics, p_value = ss.mannwhitneyu(treatment, control)
    # Compute A12
    estimate, magnitude = VD_A(treatment, control)

    # Print them
    print("Comparing: %s,%s.\n \t p-Value %s - %s \n \t A12 %f - %s " %(
             treatment_name.replace("\n", " "), control_name.replace("\n", " "),
             statistics, p_value,
             estimate, magnitude))
# LF
# AV = [0.6108745440321892, 0.6537702870466516, 0.5154796434450712, 0.5733641427206373, 0.6227785896478959, 0.6514008926700379, 0.42093543561623203, 0.5139152760758345, 0.5036128114281226, 0.5946578435561058] 
# MV = [0.6476190476190476, 0.660377358490566, 0.660377358490566, 0.6538461538461539, 0.6476190476190476, 0.6538461538461539, 0.5102040816326531, 0.6481481481481483, 0.6226415094339622, 0.6666666666666666]  
# TRV =  [0.9323308270676692, 0.9090909090909092, 0.6476190476190476, 0.660377358490566, 0.890625, 0.9558823529411764, 0.9323308270676692, 0.6476190476190476, 0.8837209302325583, 0.9402985074626865]  
# LRV = [0.9323308270676692, 0.9027777777777778, 0.6476190476190476, 0.6476190476190476, 0.8799999999999999, 0.9558823529411764, 0.9393939393939393, 0.6476190476190476, 0.890625, 0.9402985074626865] 

# TC
AV = [0.553882077799844, 0.5486595747752164, 0.5854437195224866, 0.7468260073260072, 0.44645080396348985, 0.29294958204705285, 0.6050232596467764, 0.6271187982178045, 0.5620010549807264, 0.5191754748777598]  
MV = [0.9565217391304348, 0.7333333333333334, 0.8979591836734695, 0.9166666666666666, 0.31654676258992803, 0.27329192546583847, 0.6176470588235294, 0.7586206896551724, 0.3387096774193548, 0.8518518518518519] 
TRV = [0.9777777777777777, 0.42718446601941745, 0.9777777777777777, 0.9777777777777777, 0.9387755102040816, 0.17187500000000003, 0.9565217391304348, 0.9777777777777777, 0.9777777777777777, 0.9583333333333334] 
LRV = [0.9777777777777777, 0.423076923076923, 0.9777777777777777, 0, 0.9387755102040816, 0.17120622568093388, 0.9565217391304348, 0.9565217391304348, 0.9777777777777777, 0.9565217391304348]  

treatment = TRV
treatment_name = 'treatment'
control = MV
control_name = 'control'
_log_raw_statistics(treatment, treatment_name, control, control_name)