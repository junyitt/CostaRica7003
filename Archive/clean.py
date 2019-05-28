import pandas as pd
import numpy as np

import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import xgboost
from xgboost import XGBClassifier


def clean_test(df):
    #remove ID
    df = df.drop(["Id", "idhogar"], axis = 1)
    
    ##*impute meaneduc
    df["meaneduc"] = df["meaneduc"].fillna(0)
    
    #impute v2a1 (Monthly rent payment) with mean
    df["v2a1"] = df["v2a1"].fillna(df["v2a1"].mean())

    #impute v18q1 (number of tablets household owns) with 0
    df["v18q1"] = df["v18q1"].fillna(0)

    #remove rez_esc column (Years behind in school)
    df = df.drop("rez_esc", axis = 1)

    #replace yes with 1, replace no with 0
    df[["edjefe", "edjefa"]] = df[["edjefe", "edjefa"]].apply(lambda x: x.replace("no", 0).replace("yes", 1))

    #convert string to numeric
    df["edjefe"] = pd.to_numeric(df["edjefe"])
    df["edjefa"] = pd.to_numeric(df["edjefa"])

    #create a new column called head_edu (head of household years of education), remove the two columns
    df["head_edu"] = df[["edjefe", "edjefa"]].max(axis = 1) 
    df = df.drop(["edjefe", "edjefa"], axis = 1)

    #Add dependency_rate column
    df["dependency_rate"] = (df["hogar_mayor"] + df["hogar_nin"]) / df["hogar_total"] 
    #Remove dependency column
    df = df.drop("dependency", axis = 1)
    
    df["gender"] = df["male"] #if male then 1, else 0
    df = df.drop(["male", "female"], axis = 1)

    wall_columns = list(df.filter(regex='^pared').columns)
    print(wall_columns)
    if (df[wall_columns].sum(axis = 1) == 1).all(): #check if sum of the 8 columns is 1 for all rows
        df = df.drop(["paredother"], axis = 1) #if yes, then remove 1 column. In this case, remove 'paredother'
        print("Removed 'paredother'.")

    floor_columns = list(df.filter(regex='^piso').columns)
    print(floor_columns)
    if (df[floor_columns].sum(axis = 1) == 1).all(): #check if sum of the relevant columns is 1 for all rows
        df = df.drop(["pisoother"], axis = 1) #if yes, then remove 1 column. 
        print("Removed 'pisoother'.")

    roof_columns = list(df.filter(regex='^techo').columns) 
    print(roof_columns)
    print(df[df[roof_columns].sum(axis = 1) == 0][roof_columns + ["cielorazo"]].describe()) 
    print(df[df[roof_columns].sum(axis = 1) == 1][roof_columns + ["cielorazo"]].describe()) 
    #This shows that if the 4 columns (techo*) are all 0, then 'cielorazo'is 0.
    #However, there are cases where any of the 4 columns with 1, does not imply 'cielorazo' is 1. 
    #(e.g. There are cases where the roof is zink, but the house has no ceiling)
    #Therefore, we cannot remove any of the 4 columns and 'cielorazo'
    
    water_columns = list(df.filter(regex='^abastagua').columns)
    print(water_columns)
    if (df[water_columns].sum(axis = 1) == 1).all(): #check if sum of the relevant columns is 1 for all rows
        df = df.drop(["abastaguano"], axis = 1) #if yes, then remove 1 column. 
        print("Removed 'abastaguano'.")

    electricity_columns = ["public", "planpri", "noelec", "coopele"]
    print(electricity_columns)
    (df[electricity_columns].sum(axis = 1) == 1).all() #check if sum of the relevant columns is 1 for all rows
    # Does not require removing any column
    
    toilet_columns = list(df.filter(regex='^sanitario').columns)
    print(toilet_columns)
    if (df[toilet_columns].sum(axis = 1) == 1).all(): #check if sum of the relevant columns is 1 for all rows
        df = df.drop(["sanitario6"], axis = 1) #if yes, then remove 1 column. 
        print("Removed 'sanitario6'.")
        
    energy_columns = list(df.filter(regex='^energcocinar').columns)
    print(energy_columns)
    if (df[energy_columns].sum(axis = 1) == 1).all(): #check if sum of the relevant columns is 1 for all rows
        df = df.drop(["energcocinar1"], axis = 1) #if yes, then remove 1 column. 
        print("Removed 'energcocinar1'.")

    rubbish_columns = list(df.filter(regex='^elimbasu').columns)
    print(rubbish_columns)
    if (df[rubbish_columns].sum(axis = 1) == 1).all(): #check if sum of the relevant columns is 1 for all rows
        df = df.drop(["elimbasu6"], axis = 1) #if yes, then remove 1 column. 
        print("Removed 'elimbasu6'.")

    wallc_columns = list(df.filter(regex='^epared').columns)
    print(wallc_columns)
    if (df[wallc_columns].sum(axis = 1) == 1).all(): #check if sum of the relevant columns is 1 for all rows
        df = df.drop(["epared2"], axis = 1) #if yes, then remove 1 column. 
        print("Removed 'epared2'.")
    
    roofc_columns = list(df.filter(regex='^etecho').columns)
    print(roofc_columns)
    if (df[roofc_columns].sum(axis = 1) == 1).all(): #check if sum of the relevant columns is 1 for all rows
        df = df.drop(["etecho2"], axis = 1) #if yes, then remove 1 column. 
        print("Removed 'etecho2'.")

    floorc_columns = list(df.filter(regex='^eviv').columns)
    print(floorc_columns)
    if (df[floorc_columns].sum(axis = 1) == 1).all(): #check if sum of the relevant columns is 1 for all rows
        df = df.drop(["eviv2"], axis = 1) #if yes, then remove 1 column. 
        print("Removed 'eviv2'.")

    civil_columns = list(df.filter(regex='^estadocivil').columns)
    print(civil_columns)
    if (df[civil_columns].sum(axis = 1) == 1).all(): #check if sum of the relevant columns is 1 for all rows
        df = df.drop(["estadocivil7"], axis = 1) #if yes, then remove 1 column. 
        print("Removed 'estadocivil7'.")

    household_columns = list(df.filter(regex='^parentesco').columns)
    print(household_columns)
    if (df[household_columns].sum(axis = 1) == 1).all(): #check if sum of the relevant columns is 1 for all rows
        df = df.drop(["parentesco12"], axis = 1) #if yes, then remove 1 column. 
        print("Removed 'parentesco12'.")

    edulevel_columns = list(df.filter(regex='^instlevel').columns)
    print(edulevel_columns)
    if (df[edulevel_columns].sum(axis = 1) == 1).all(): #check if sum of the relevant columns is 1 for all rows
        df = df.drop(["instlevel9"], axis = 1) #if yes, then remove 1 column. 
        print("Removed 'instlevel9'.")
    else:
        print(df[edulevel_columns].sum(axis = 1).describe())
    
    houseown_columns = list(df.filter(regex='^tipovivi').columns)
    print(houseown_columns)
    if (df[houseown_columns].sum(axis = 1) == 1).all(): #check if sum of the relevant columns is 1 for all rows
        df = df.drop(["tipovivi5"], axis = 1) #if yes, then remove 1 column. 
        print("Removed 'tipovivi5'.")

    region_columns = list(df.filter(regex='^lugar').columns)
    print(region_columns)
    if (df[region_columns].sum(axis = 1) == 1).all(): #check if sum of the relevant columns is 1 for all rows
        df = df.drop(["lugar6"], axis = 1) #if yes, then remove 1 column. 
        print("Removed 'lugar6'.")

    settlement_columns = list(df.filter(regex='^area').columns)
    print(settlement_columns)
    if (df[settlement_columns].sum(axis = 1) == 1).all(): #check if sum of the relevant columns is 1 for all rows
        df = df.drop(["area2"], axis = 1) #if yes, then remove 1 column. 
        print("Removed 'area2'.")
        
    # hhsize, r4h1, r4h2, r4h3, r4m1, r4m2, r4m3, r4t1, r4t2, r4t3, tamhog, tamviv
    # include: r4h1, r4h2, r4m1, r4m2, tamhog
    # remove: hhsize, r4h3, r4m3, r4t1, r4t2, r4t3, tamviv
    df = df.drop(["hhsize", "r4h3", "r4m3", "r4t1", "r4t2", "r4t3", "tamviv"], axis = 1)

    ## dependency_rate were calculated previously, include only hogar_nin
    ## remove hogar_adul, hogar_mayor, hogar_total
    df = df.drop(["hogar_adul", "hogar_mayor", "hogar_total"], axis = 1)

    #remove mobilephone (implied by qmobilephone)
    df = df.drop(["mobilephone"], axis = 1)
    
    # Remove columns that are square of another column
    square_columns = list(df.filter(like='SQ').columns) + ["agesq"]
    df = df.drop(square_columns, axis = 1)

    scaler = preprocessing.StandardScaler().fit(df[["v2a1"]].iloc[0:int(0.6*df.shape[0])])
    df["v2a1"] = scaler.transform(df[["v2a1"]])   

    return df




def clean_test_2(df):
    #remove ID
    df = df.drop(["Id", "idhogar"], axis = 1)
    
    ##*impute meaneduc
    df["meaneduc"] = df["meaneduc"].fillna(0)
    
    #impute v2a1 (Monthly rent payment) with mean
    df["v2a1"] = df["v2a1"].fillna(df["v2a1"].mean())

    #impute v18q1 (number of tablets household owns) with 0
    df["v18q1"] = df["v18q1"].fillna(0)

    #remove rez_esc column (Years behind in school)
    df = df.drop("rez_esc", axis = 1)

    #replace yes with 1, replace no with 0
    df[["edjefe", "edjefa"]] = df[["edjefe", "edjefa"]].apply(lambda x: x.replace("no", 0).replace("yes", 1))

    #convert string to numeric
    df["edjefe"] = pd.to_numeric(df["edjefe"])
    df["edjefa"] = pd.to_numeric(df["edjefa"])

    #create a new column called head_edu (head of household years of education), remove the two columns
    df["head_edu"] = df[["edjefe", "edjefa"]].max(axis = 1) 
    df = df.drop(["edjefe", "edjefa"], axis = 1)

    #Add dependency_rate column
    df["dependency_rate"] = (df["hogar_mayor"] + df["hogar_nin"]) / df["hogar_total"] 
    #Remove dependency column
    df = df.drop("dependency", axis = 1)
    
    df["gender"] = df["male"] #if male then 1, else 0
    df = df.drop(["male", "female"], axis = 1)

    # hhsize, r4h1, r4h2, r4h3, r4m1, r4m2, r4m3, r4t1, r4t2, r4t3, tamhog, tamviv
    # include: r4h1, r4h2, r4m1, r4m2, tamhog
    # remove: hhsize, r4h3, r4m3, r4t1, r4t2, r4t3, tamviv
    df = df.drop(["hhsize", "r4h3", "r4m3", "r4t1", "r4t2", "r4t3", "tamviv"], axis = 1)

    ## dependency_rate were calculated previously, include only hogar_nin
    ## remove hogar_adul, hogar_mayor, hogar_total
    df = df.drop(["hogar_adul", "hogar_mayor", "hogar_total"], axis = 1)

    #remove mobilephone (implied by qmobilephone)
    df = df.drop(["mobilephone"], axis = 1)
    
    # Remove columns that are square of another column
    square_columns = list(df.filter(like='SQ').columns) + ["agesq"]
    df = df.drop(square_columns, axis = 1)

    scaler = preprocessing.StandardScaler().fit(df[["v2a1"]].iloc[0:int(0.6*df.shape[0])])
    df["v2a1"] = scaler.transform(df[["v2a1"]])   

    return df