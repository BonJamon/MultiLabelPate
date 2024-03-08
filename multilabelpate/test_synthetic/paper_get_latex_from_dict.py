import pickle
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import FormatStrFormatter

#COlORS
#f1s
DARKRED = "#a00000"
MEDRED = "#c46666"
LIGHTRED = "#d8a6a6"
#Divergent
YELLOW = "#e9c716"
BLUE = "#1a80bb" #also percentage 
TEAL = "#50ad9f" #Also privacy cost

########BLUEPRINT AGGREGATION PERFORMANCE##########
if(False):
    with open(os.path.join("results","paper_evaluateAggregation_final"), "rb") as f:
        result = pickle.load(f)
    filtered_keys = ["single","avg","aggr adapt","aggr fixed", "aggr adapt t-approx","aggr fixed t-approx"]
    filtered_keys_T = ["aggr T=5","aggr T=10","aggr T=15", "aggr adapt"]
    head_str = "data"
    result_list = list(result.items())
    for i in range(len(result_list)):
        data = result_list[i][0]
        print_str2 = "$"+str(data)+"$"
        for key in result_list[i][1].keys():
            if key in filtered_keys:
                if i==0:
                    head_str += "&" + key
                print_str2 += "&"
                val = result_list[i][1].get(key)
                print_str2 += "{0:0.2f}".format(round(val[0], 2))+"/"+"{0:0.2f}".format(round(val[1], 2))+"/"+"{0:0.2f}".format(round(val[2], 2))
        print_str2 += "\\\\"
        if i==0:
            head_str += "\\\\"
            print(head_str)
        print(print_str2)


#########BLUEPRINT PRIVACY PERFORMANCE############
if(False):
    with open(os.path.join("results","paper_evaluatePrivacy"), "rb") as f:
        result = pickle.load(f)

    filtered_keys_p = ["results adapt", "results fixed"]
    N_per_teacher = 1000
    N=300
    i=1
    dataset = str(N)+"_"+str(i)
    head_str = "noise"
    result_dataset = result[dataset]
    result_list = list(result_dataset.items())
    for i in range(len(result_list)):
        noise = result_list[i][0]
        print_str2 = str(noise)
        for key in result_list[i][1].keys():
            if key in filtered_keys_p:
                if i==0:
                    head_str += "&" + key
                print_str2 += "&"
                val = result_list[i][1].get(key)
                print_str2 += "{0:0.2f}".format(round(val[0], 2))+"/"+"{0:0.2f}".format(round(val[1], 2))+"/"+"{0:0.2f}".format(round(val[2], 2))+"_dp:"+"{0:0.4f}".format(round(val[3]/(N*N_per_teacher), 4))+"_%:"+"{0:0.2f}".format(round(val[4], 2))
        print_str2 += "\\\\"
        if i==0:
            head_str += "\\\\"
            print(head_str)
        print(print_str2)

########Blueprint Confidence Check###############
if(False):
    font = {'family' : 'arial',
        'weight' : 'normal',
        'size'   : 12}

    matplotlib.rc('font', **font)
    with open(os.path.join("results","paper_analyseConfidenceCheck_thresholdBased"), "rb") as f:
        result = pickle.load(f)
    n_points = 300*1000

    filtered_keys_p = ["per points","performance adapt","dp cost adapt","performance fixed","dp cost fixed"]
    dataset = "300_4"
    head_str = "n & noises"
    result_dataset = result[dataset]
    result_list = list(result_dataset.items())

    #initalize plots
    n_ns = 3
    n_noises = 3
    figure, axis_f1 = plt.subplots(n_ns,n_noises, sharey=True)

    axis_dps = []
    for i in range(n_ns):
        inner_list = []
        for j in range(n_noises):
            inner_list.append(axis_f1[i,j].twinx())
        axis_dps.append(inner_list)

    for i in range(len(result_list)):
        noises = result_list[i][0]
        result_list_noises = list(result_list[i][1].items())
        for j in range(len(result_list_noises)):
            n = result_list_noises[j][0]
            result_list_confthr = list(result_list_noises[j][1].items())

            f1s = np.zeros(len(result_list_confthr))
            dp_costs = np.zeros(len(result_list_confthr))
            per_points = np.zeros(len(result_list_confthr))
            conf_thresholds = np.ones(len(result_list_confthr)+1) *1.3
            for k in range(len(result_list_confthr)):
                conf_interval = result_list_confthr[k][0]
                conf_thresholds[k] = float(conf_interval[:3])
                for key in result_list_confthr[j][1].keys():
                    if key == "per points":
                        per_points[k] = result_list_confthr[k][1].get(key)
                    if key == "performance fixed":
                        f1s[k] = result_list_confthr[k][1].get(key)[2]
                    if key == "dp cost fixed":
                        dp_costs[k] = result_list_confthr[k][1].get(key)

            center_bins = np.zeros(len(conf_thresholds)-1)
            for l in range(len(conf_thresholds)-1):
                center_bins[l] = (conf_thresholds[l]+conf_thresholds[l+1]) / 2
            #get dp_cost per point
            points_per_conf_interval = n_points * per_points
            dp_costs_normalized = np.divide(dp_costs, points_per_conf_interval)

            lns1 = axis_f1[i,j].bar(center_bins, height=per_points, width=0.1, label="%points")
            lns2 = axis_f1[i,j].plot(center_bins[f1s>0], f1s[f1s>0], color="r", label="f1-score")
            axis_f1[i,j].set_ylim(0,1)
            axis_f1[i,j].set_xlim(0,1.4)
            lns3 = axis_dps[i][j].plot(center_bins, dp_costs_normalized, color="g", label="privacy cost")
            axis_dps[i][j].set_ylim(0,0.1)
            axis_dps[i][j].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            #axis_f1[i,j].get_yaxis().set_visible(False)
            if j > 0:
                axis_f1[i,j].get_yaxis().set_visible(False)
            if j < 2:
                axis_dps[i][j].get_yaxis().set_visible(False)
            if i==len(result_list_noises)-1:
                n = re.findall(r'\d+', n)[0]
                axis_f1[i,j].set_xlabel("n="+str(n))
            if j==0:
                noise_c,noise = re.findall(r'\d+', noises)
                axis_f1[i,j].set_ylabel(r"$\sigma_1$"+"="+str(noise_c)+","+r"$\sigma_2$"+"="+str(noise))
                #if i==0:
                 #   axis_f1[i,j].set_ylabel("f1")
            if j==2 and i==0:
                #axis_dps[i][j].set_ylabel(r"$\epsilon$")
                handles, labels = axis_f1[i,j].get_legend_handles_labels()
                handle_dp, label_dp = axis_dps[i][j].get_legend_handles_labels()
                handles.append(handle_dp[0])
                labels.append(label_dp[0])
                figure.legend(handles, labels, bbox_to_anchor=[0.98,0.88])

    figure.supxlabel('confidence threshold')
    plt.show()
            

#Subset: n=5, noise_c=9600, noise=6400
if(False):
    font = {'family' : 'arial',
        'weight' : 'normal',
        'size'   : 12}

    matplotlib.rc('font', **font)
    with open(os.path.join("results","paper_analyseConfidenceCheck_thresholdBased"), "rb") as f:
        result = pickle.load(f)
    n_points = 300*1000

    filtered_keys_p = ["per points","performance adapt","dp cost adapt","performance fixed","dp cost fixed"]
    dataset = "300_1"
    result_dataset = result[dataset]
    result_list = list(result_dataset.items())
    
    j=1
    i=2
    noises = result_list[i][0]
    result_list_noises = list(result_list[i][1].items())
    n = result_list_noises[j][0]
    result_list_confthr = list(result_list_noises[j][1].items())

    f1s = np.zeros(len(result_list_confthr))
    dp_costs = np.zeros(len(result_list_confthr))
    per_points = np.zeros(len(result_list_confthr))
    conf_thresholds = np.ones(len(result_list_confthr)+1) *1.3
    for k in range(len(result_list_confthr)):
        conf_interval = result_list_confthr[k][0]
        conf_thresholds[k] = float(conf_interval[:3])
        for key in result_list_confthr[j][1].keys():
            if key == "per points":
                per_points[k] = result_list_confthr[k][1].get(key)
            if key == "performance fixed":
                f1s[k] = result_list_confthr[k][1].get(key)[2]
            if key == "dp cost fixed":
                dp_costs[k] = result_list_confthr[k][1].get(key)

    center_bins = np.zeros(len(conf_thresholds)-1)
    for l in range(len(conf_thresholds)-1):
        center_bins[l] = (conf_thresholds[l]+conf_thresholds[l+1]) / 2
    #get dp_cost per point
    points_per_conf_interval = n_points * per_points
    dp_costs_normalized = np.divide(dp_costs, points_per_conf_interval)

    center_bins = conf_thresholds[:-1]
    print(center_bins)

    #plot
    figure, axis_f1 = plt.subplots(1,1)
    axis_dp = axis_f1.twinx()

    lns1 = axis_f1.bar(center_bins, height=per_points, width=0.08, label="%points", color=BLUE)
    lns2 = axis_f1.plot(center_bins[f1s>0], f1s[f1s>0], color=DARKRED, label="f1-score")
    axis_f1.set_ylim(0,1)
    axis_f1.set_xlim(-0.05,1.2)
    lns3 = axis_dp.plot(center_bins, dp_costs_normalized, color=TEAL, label="privacy cost")
    #axis_dp.set_ylim(0,0.07)


    #label axis
    axis_f1.set_ylabel("f1")
    axis_dp.set_ylabel(r"$\epsilon$")
    axis_f1.set_xlabel(r"$T_c$"+" / M")
    #make legend
    handles, labels = axis_f1.get_legend_handles_labels()
    handle_dp, label_dp = axis_dp.get_legend_handles_labels()
    handles.append(handle_dp[0])
    labels.append(label_dp[0])
    figure.legend(handles, labels, bbox_to_anchor=[0.90,0.88])

    plt.show()
#Privacy Cost Distribution single threshold
if(False):
    with open(os.path.join("results","paper_analyseConfidenceCheck_chosenThreshold"), "rb") as f:
        result = pickle.load(f)
    n_points = 300*1000

    filtered_keys_p = ["per points","performance adapt","dp cost adapt aggr","dp cost adapt check"]
    dataset = "300_1"
    head_str = "n & noises"
    result_dataset = result[dataset]
    result_list = list(result_dataset.items())

    #initalize plots
    n_ns = 3
    n_noises = 3
    figure, axis_f1 = plt.subplots(1,1)
    axis_dp = axis_f1.twinx()

    f1s = np.zeros(len(result_list))
    dp_costs_aggr = np.zeros(len(result_list))
    dp_costs_check = np.zeros(len(result_list))
    per_points = np.zeros(len(result_list))
    conf_thresholds = np.ones(len(result_list)+1) *1.3
    for k in range(len(result_list)):
        conf_interval = result_list[k][0]
        conf_thresholds[k] = float(conf_interval[:3])
        for key in result_list[k][1].keys():
            if key == "per points":
                per_points[k] = result_list[k][1].get(key)
            if key == "performance adapt":
                f1s[k] = result_list[k][1].get(key)[2]
            if key == "dp cost adapt aggr":
                dp_costs_aggr[k] = result_list[k][1].get(key)
            if key == "dp cost adapt check":
                dp_costs_check[k] = result_list[k][1].get(key)

    center_bins = np.zeros(len(conf_thresholds)-1)
    for l in range(len(conf_thresholds)-1):
        center_bins[l] = (conf_thresholds[l]+conf_thresholds[l+1]) / 2
    #get dp_cost per point
    points_per_conf_interval = n_points * per_points
    dp_costs_aggr_normalized = np.divide(dp_costs_aggr, points_per_conf_interval)
    dp_costs_check_normalized = np.divide(dp_costs_check, points_per_conf_interval)

    axis_f1.bar(center_bins, height=per_points, width=0.1)
    axis_f1.plot(center_bins, f1s, color="r", label="f1")
    axis_f1.set_ylim(0,1)
    axis_dp.plot(center_bins, dp_costs_aggr_normalized, color="limegreen", label="dp aggr")
    axis_dp.plot(center_bins, dp_costs_check_normalized, color="darkgreen", label="dp check")
    axis_dp.set_ylim(0,0.002)
    #axis_dp.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    plt.legend()
    plt.show()





#################Blueprint interactive analysis############################
if(False):
    with open(os.path.join("results","paper_analyseInteractiveCheck_updatedPrivacyBound"), "rb") as f:
        result = pickle.load(f)
    n_points = 300*1000

    filtered_keys_p = ["perPointsAggr adapt","perPointsReinforced adapt","pre/rec/f1 aggr adapt","pre/rec/f1 student adapt","dp adapt"]
    dataset = "300_1"
    result_dataset = result[dataset]
    result_list = list(result_dataset.items())

    #initalize plots
    n_ns = 3
    n_confthr = 3
    figure, axis_f1 = plt.subplots(n_ns,n_confthr)

    axis_dps = []
    for i in range(n_ns):
        inner_list = []
        for j in range(n_confthr):
            inner_list.append(axis_f1[i,j].twinx())
        axis_dps.append(inner_list)

    for i in range(len(result_list)):
        n = result_list[i][0]
        result_list_studentConfthr = list(result_list[i][1].items())
        for j in range(len(result_list_studentConfthr)):
            noises = result_list_studentConfthr[j][0]
            print_str = noises + str(n)
            result_list_confthr = list(result_list_studentConfthr[j][1].items())

            f1s_aggr = np.zeros(len(result_list_confthr))
            f1s_reinforced = np.zeros(len(result_list_confthr))
            dp_costs = np.zeros(len(result_list_confthr))
            per_points_aggr = np.zeros(len(result_list_confthr))
            per_points_reinforced = np.zeros(len(result_list_confthr))
            conf_thresholds = np.ones(len(result_list_confthr)+1) *0.6
            for k in range(len(result_list_confthr)):
                conf_interval = result_list_confthr[k][0]
                print_str2 = print_str + "&"
                print_str2 += conf_interval
                idx = re.search("_", conf_interval).start()
                conf_thresholds[k] = float(conf_interval[:idx])
                for key in result_list_confthr[j][1].keys():
                    if key == "perPointsAggr fixed":
                        per_points_aggr[k] = result_list_confthr[k][1].get(key)
                    if key == "perPointsReinforced fixed":
                        per_points_reinforced[k] = result_list_confthr[k][1].get(key)
                    if key == "pre/rec/f1 aggr fixed":
                        f1s_aggr[k] = result_list_confthr[k][1].get(key)[2]
                    if key == "pre/rec/f1 student fixed":
                        f1s_reinforced[k] = result_list_confthr[k][1].get(key)[2]
                    if key == "dp fixed":
                        dp_costs[k] = result_list_confthr[k][1].get(key)

            center_bins = np.zeros(len(conf_thresholds)-1)
            for l in range(len(conf_thresholds)-1):
                center_bins[l] = (conf_thresholds[l]+conf_thresholds[l+1]) / 2
            #get dp_cost per point
            points_per_conf_interval = n_points * (per_points_aggr + per_points_reinforced)
            dp_costs_normalized = np.divide(dp_costs, points_per_conf_interval)
            #normalize counts
            per_points_aggr = per_points_aggr / 1000.0
            per_points_reinforced = per_points_reinforced / 1000.0

            #plot
            axis_f1[i,j].bar(center_bins, height=per_points_aggr, width=0.1, label="aggr")
            axis_f1[i,j].bar(center_bins, height=per_points_reinforced, bottom=per_points_aggr, width=0.1,label="reinforced")
            axis_f1[i,j].plot(center_bins[f1s_aggr>0], f1s_aggr[f1s_aggr>0], color="blue", label="f1 aggr")
            axis_f1[i,j].plot(center_bins[f1s_reinforced>0], f1s_reinforced[f1s_reinforced>0], color="orange", label="f1 reinforced")
            axis_f1[i,j].set_ylim(0,1)
            axis_dps[i][j].plot(center_bins, dp_costs_normalized, color="g")
            axis_dps[i][j].set_ylim(0,2.0e-7)
            #clean up plot
            if j > 0:
                #axis_f1[i,j].tick_params(axis="y",colors="w")
                axis_f1[i,j].get_yaxis().set_visible(False)
            if j < 2:
                #axis_dps[i][j].tick_params(axis="y", colors="w")
                axis_dps[i][j].get_yaxis().set_visible(False)
            if i==2:
                axis_f1[i,j].set_xlabel(str(noises))
            if j==0:
                axis_f1[i,j].set_ylabel(str(n))
    plt.show()
#################BLUEPRINT STUDENT EVALUATION############################
if(False):
    with open(os.path.join("results","paper_get_student_performance_interactive_all_noises"), "rb") as f:
        result = pickle.load(f)
    dataset = "300_1"
    head_str = "noise_c,noise & N_kept & pre/rec/f1 adapt & dp adapt & pre/rec/f1 fixed & dp fixed \\\\"
    result_dataset = result[dataset]
    result_list = list(result_dataset.items())
    for i in range(len(result_list)):
        noise = result_list[i][0]
        result_noise = result_list[i][1]
        result_list_noise=list(result_noise.items())
        for j in range(len(result_list_noise)):
            N_kept = result_list_noise[j][0]
            print_str2 = str(noise)+ "&"+str(N_kept)
            result_N_kept = result_list_noise[j][1]
            for key in result_N_kept.keys():
                val = result_N_kept.get(key)
                print_str2 += "&" + "{0:0.2f}".format(round(val[0], 2))+"/"+"{0:0.2f}".format(round(val[1], 2))+"/"+"{0:0.2f}".format(round(val[2], 2))
                print_str2 += "&" + "{0:0.2f}".format(round(val[3], 2))
            print_str2 += "\\\\"
            if i==0 and j==0:
                print(head_str)
            print(print_str2)



###########BLUEPRINT COMPARATIVE SCATTER PLOT#######################
if(False):
    filenames = ["paper_get_student_performance_interactive_all_noises_updatedPrivacyBound","paper_get_student_performance_confident_all_noises","paper_get_student_performance_random_all_noises"]
    versions = ["interactive", "confident", "basic"]
    baselines = [0.70,0.68,0.69,0.73,0.68]
    dataset = 1
    baseline = baselines[dataset]
    colours = [BLUE, TEAL, YELLOW]
    fig,ax = plt.subplots()
    ax.set_ylim(0.35,0.72)
    ax.set_xlim(0,23)
    for k in range(len(filenames)):
        filename = filenames[k]
        version = versions[k]
        color = colours[k]
        with open(os.path.join("results",filename), "rb") as f:
            result = pickle.load(f)
        #init
        dps = []
        f1s = []
        #read data
        result_dataset = result["300_"+str(dataset)]
        result_list = list(result_dataset.items())
        for i in range(len(result_list)):
            noise = result_list[i][0]
            result_noise = result_list[i][1]
            result_list_noise=list(result_noise.items())
            for j in range(len(result_list_noise)):
                N_kept = result_list_noise[j][0]
                result_N_kept = result_list_noise[j][1]
                for key in result_N_kept.keys():
                    if key == "results_fixed":
                        val = result_N_kept.get(key)
                        f1s.append(val[2])
                        dps.append(val[3])
        ax.scatter(x=dps, y=f1s, c=color,label=version)
    ax.hlines(y=baseline, xmin=0, xmax=23, label="single teacher", colors=DARKRED)
    ax.hlines(y=0.64, xmin=0, xmax=23, label="aggregation: no noise", colors=MEDRED)
    ax.hlines(y=0.63, xmin=0, xmax=23, label="aggregation: no noise and no "+r"$\tau$"+"-approx.", colors=LIGHTRED)
    #ax.hlines(y=0.39, xmin=0, xmax=23, label="random predictor", colors="lightcoral")
    ax.legend()
    ax.grid(True)

    plt.xlabel("privacy cost")
    plt.ylabel("f1 score")
    plt.show()
                    

###########NUS-Wide Avg teacher performances###############
if(True):
    maps = [0.53,0.49,0.45,0.40,0.33]
    f1s = [0.47,0.45,0.42,0.37,0.31]
    accs = [0.97,0.97,0.97,0.97,0.97]
    pres = [0.49,0.50,0.45,0.41,0.34]
    recs = [0.58,0.52,0.49,0.45,0.40]

    teachers = [1,2,4,8,16]
    
    plt.plot(teachers, accs, label="accuracy", color=YELLOW)
    plt.plot(teachers, maps, label="map", color=BLUE)
    plt.plot(teachers, f1s, label="f1", color=DARKRED)
    #plt.plot(teachers, recs, label="recall")
    #plt.plot(teachers, pres, label="precision")
    plt.xlabel("M")
    

    plt.legend()
    plt.show()

    plt.imsave("avgplot",)