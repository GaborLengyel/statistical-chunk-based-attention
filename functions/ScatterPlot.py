def ScatterForCorrelation(data,
                        SpreadOfX = 0.1,
                        YError = 'CI',
                        NoStd = 1,
                        plotsize = [15,15],
                        axeslimitX = [],
                        axeslimitY = [],
                        axisLabels = [],
                        SameAxisLabel = True,
                        ConditionLabels = [],
                        SameConditionLabels = True,
                        SubplotTitles = [],
                        SameSubplotTitles = True,
                        plotTitle = [],
                        ThresValue = [],
                        SameThresValue = True,
                        RegressionLine = True,
                        AxisTicks = [],
                        SameAxisTicks = True,
                        SaveFigName = [],
                        Outliers = [],
                        SameErrorEllipseColor = True,
                        ErrorEllipse = {'color':[],'alpha':0.2},
                        Ticks = {'width':2, 'length':6},
                        axisTouching = False,
                        ErrorBar = {'ErrDist':[], 'ErrSize':1, 'ErrWid':6, 'sizedots':10, 'ErrColor':[], 'DotsColor':[],'OutlierDotsColor':'r','RegLineWid':2},
                        titleFont = {'fontname':'Arial', 'size':'28', 'color':'black', 'weight':'normal','verticalalignment':'bottom'},
                        axisFont = {'fontname':'Arial', 'size':'24'},
                        axisWidth = 3,
                        LegendPos = {'LegendPosition':'upper left'},
                        FigureLayout = [1,1,1,0.95]):

    # imports
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats as sc
    from matplotlib.patches import Ellipse

    # parameters for the figure:
    ColSubplots = len(data)
    RowSubplots = len(data[0])
    NScatter = len(data[0][0])

    # computing the position and length of the error bars
    ErrPosXY = np.zeros((ColSubplots,RowSubplots,NScatter,2))
    ErrLenXY = np.zeros((ColSubplots,RowSubplots,NScatter,2))
    for Row in range(RowSubplots):
        for Col in range(ColSubplots):
            SortedMeanXY = np.zeros((NScatter))
            AllXY = np.empty((2,0))
            for Scatter in range(len(data[Col][Row])):

                Ys = data[Col][Row][Scatter][1]
                Xs = data[Col][Row][Scatter][0]
                SortedMeanXY[Scatter] = np.array([np.mean(Ys)])
                AllXY = np.hstack((AllXY,[Xs,Ys]))

            SortedMeanXY = np.argsort(SortedMeanXY, axis = 0)
            for i in range(SortedMeanXY.shape[0]-1,-1,-1):
                ErrPosXY[Col,Row,SortedMeanXY[i],1] = np.amin(AllXY[1,:])-(np.ptp(AllXY[1,:]*ErrorBar['ErrDist'][0][i]))
                ErrPosXY[Col,Row,SortedMeanXY[i],0] = np.amin(AllXY[0,:])-(np.ptp(AllXY[0,:]*ErrorBar['ErrDist'][1][i]))
                ErrLenXY[Col,Row,SortedMeanXY[i],1] = (np.ptp(AllXY[1,:]*ErrorBar['ErrSize']*0.1)) # TODO make it less stupit
                ErrLenXY[Col,Row,SortedMeanXY[i],0] = (np.ptp(AllXY[0,:]*ErrorBar['ErrSize']*0.1))

    # for the regression line
    if RegressionLine and len(RegressionLine)==1:
        RegressionLine = np.repeat(RegressionLine,ColSubplots*RowSubplots)

    # Figure
    fig, ax = plt.subplots(RowSubplots,ColSubplots, figsize=(plotsize[0], plotsize[1]), squeeze=False)
    
    ax = ax.ravel()

    SubPlot = -1
    Scatteridx = -1
    for Row in range(RowSubplots):
        for Col in range(ColSubplots):

            SubPlot = SubPlot+1
            for Scatter in range(len(data[Col][Row])):

                Scatteridx = Scatteridx+1
                # data:
                Ys = data[Col][Row][Scatter][1]
                Xs = data[Col][Row][Scatter][0]
                BarsY = ErrPosXY[Col,Row,Scatter,0]
                BarsX = ErrPosXY[Col,Row,Scatter,1]
                ErrLenY = ErrLenXY[Col,Row,Scatter,0]
                ErrLenX = ErrLenXY[Col,Row,Scatter,1]
                # Error intervals For Ys
                # 95% Conf.Intervals
                if YError == 'CI':
                    LowerErrorY = sc.t.interval(0.95, len(Ys)-1, loc=np.mean(Ys), scale=sc.sem(Ys))[0]
                    UpperErrorY = sc.t.interval(0.95, len(Ys)-1, loc=np.mean(Ys), scale=sc.sem(Ys))[1]
                    err_ecl = 1.96*(1.0/np.sqrt(len(Ys)))
                # standard deviation
                elif YError == 'STD':
                    LowerErrorY = np.mean(Ys)-np.std(Ys)
                    UpperErrorY = np.mean(Ys)+np.std(Ys)
                    err_ecl = 1.0
                # SEM
                elif YError == 'SEM':
                    LowerErrorY = np.mean(Ys)-sc.sem(Ys)
                    UpperErrorY = np.mean(Ys)+sc.sem(Ys)
                    err_ecl = 1.0/np.sqrt(len(Ys))
                # Error intervals For Xs
                # 95% Conf.Intervals
                if YError == 'CI':
                    LowerErrorX = sc.t.interval(0.95, len(Xs)-1, loc=np.mean(Xs), scale=sc.sem(Xs))[0]
                    UpperErrorX = sc.t.interval(0.95, len(Xs)-1, loc=np.mean(Xs), scale=sc.sem(Xs))[1]
                # standard deviation
                elif YError == 'STD':
                    LowerErrorX = np.mean(Xs)-np.std(Xs)
                    UpperErrorX = np.mean(Xs)+np.std(Xs)
                # SEM
                elif YError == 'SEM':
                    LowerErrorX = np.mean(Xs)-sc.sem(Xs)
                    UpperErrorX = np.mean(Xs)+sc.sem(Xs)

                # Error ellipse
                points=np.stack((Xs, Ys))
                cov = np.cov(points)
                # central point of the error ellipse
                #Thresholds
                pos=[np.mean(Xs),np.mean(Ys)]
                # for the angle we need the eigenvectors of the covariance matrix
                w,v=np.linalg.eig(cov)
                # We pick the largest eigen value
                order = w.argsort()[::-1]
                w=w[order]
                v=v[:,order]
                # we compute the angle towards the eigen vector with the largest eigen value
                theta = np.degrees(np.arctan(v[1,0]/v[0,0]))
                thetar = np.arctan(v[1,0]/v[0,0])
                # Compute the width and height of the ellipse based on the eigen values (ie the length of the vectors)
                if NoStd==False:
                    NoStd = err_ecl
                width, height = 2 * NoStd * np.sqrt(w)
                # making the ellipse
                ellip = Ellipse(xy=pos, width=width, height=height, angle=theta)
                ellip.set_alpha(ErrorEllipse['alpha'])                
                if SameErrorEllipseColor:
                    ellip.set_facecolor(ErrorEllipse['color'][Scatter])
                else:
                    ellip.set_facecolor(ErrorEllipse['color'][SubPlot][Scatter])
                    
                # computing regression lines
                if RegressionLine[SubPlot]:
                    xT=np.stack((np.ones(len(Xs)), Xs), axis=-1)
                    reg=np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(xT),xT)),np.transpose(xT)),Ys)


                # Scatter plot
                if ConditionLabels:
                    if SameConditionLabels:
                        Label = ConditionLabels[Scatter]
                    else:
                        Label = ConditionLabels[SubPlot][Scatter]
                else:
                    Label = []
                ax[SubPlot].scatter(Xs, Ys, s=ErrorBar['sizedots'], c=ErrorBar['DotsColor'][Scatter], marker="o", label=Label)
                # error ellipse
                ax[SubPlot].add_artist(ellip)
                # Error bar for Ys:
                ax[SubPlot].plot([BarsY,BarsY],[LowerErrorY,UpperErrorY], ErrorBar['ErrColor'][Scatter]+'-', linewidth=ErrorBar['ErrWid'])
                ax[SubPlot].plot([BarsY-ErrLenY,BarsY+ErrLenY],[UpperErrorY,UpperErrorY], ErrorBar['ErrColor'][Scatter]+'-', linewidth=ErrorBar['ErrWid'])
                ax[SubPlot].plot([BarsY-ErrLenY,BarsY+ErrLenY],[LowerErrorY,LowerErrorY], ErrorBar['ErrColor'][Scatter]+'-', linewidth=ErrorBar['ErrWid'])
                ax[SubPlot].plot(BarsY,np.mean(Ys),ErrorBar['ErrColor'][Scatter]+'o', markersize=ErrorBar['sizeMean'])
                # Error bar for Xs:
                ax[SubPlot].plot([LowerErrorX,UpperErrorX],[BarsX,BarsX], ErrorBar['ErrColor'][Scatter]+'-', linewidth=ErrorBar['ErrWid'])
                ax[SubPlot].plot([UpperErrorX,UpperErrorX],[BarsX-ErrLenX,BarsX+ErrLenX], ErrorBar['ErrColor'][Scatter]+'-', linewidth=ErrorBar['ErrWid'])
                ax[SubPlot].plot([LowerErrorX,LowerErrorX],[BarsX-ErrLenX,BarsX+ErrLenX], ErrorBar['ErrColor'][Scatter]+'-', linewidth=ErrorBar['ErrWid'])
                ax[SubPlot].plot(np.mean(Xs),BarsX,ErrorBar['ErrColor'][Scatter]+'o', markersize=ErrorBar['sizeMean'])
                # regression line
                if RegressionLine[SubPlot]:
                    ax[SubPlot].plot([BarsY,max(Xs)+np.std(Xs)],[reg[1]*(BarsY)+reg[0],reg[1]*(max(Xs)+np.std(Xs))+reg[0]],ErrorBar['ErrColor'][Scatter]+'-', linewidth=ErrorBar['RegLineWid'])
                # Large point for the mean values
                ax[SubPlot].plot(np.mean(Xs),np.mean(Ys),ErrorBar['ErrColor'][Scatter]+'o', markersize=ErrorBar['sizeMean']*0.75)
                # zero effect lines
                if ThresValue:
                    if SameThresValue:
                        if axeslimitX and axeslimitY:
                            ax[SubPlot].plot([axeslimitX[0][SubPlot], axeslimitX[1][SubPlot]], [ThresValue[1],ThresValue[1]], "k--", linewidth=ErrorBar['ErrWid']/2)
                            ax[SubPlot].plot([ThresValue[0],ThresValue[0]], [axeslimitY[0][SubPlot], axeslimitY[1][SubPlot]], "k--", linewidth=ErrorBar['ErrWid']/2)
                        else:
                            ax[SubPlot].plot([min(Xs), max(Xs)], [ThresValue[1],ThresValue[1]], "k--", linewidth=ErrorBar['ErrWid']/2)
                            ax[SubPlot].plot([ThresValue[0],ThresValue[0]], [min(Ys), max(Ys)], "k--", linewidth=ErrorBar['ErrWid']/2)
                    else:
                        if axeslimitX and axeslimitY:
                            ax[SubPlot].plot([axeslimitX[0][SubPlot], axeslimitX[1][SubPlot]], [ThresValue[SubPlot][1],ThresValue[SubPlot][1]], "k--", linewidth=ErrorBar['ErrWid']/2)
                            ax[SubPlot].plot([ThresValue[SubPlot][0],ThresValue[SubPlot][0]], [axeslimitY[0][SubPlot], axeslimitY[1][SubPlot]], "k--", linewidth=ErrorBar['ErrWid']/2)
                        else:
                            ax[SubPlot].plot([min(Xs), max(Xs)], [ThresValue[SubPlot][1],ThresValue[SubPlot][1]], "k--", linewidth=ErrorBar['ErrWid']/2)
                            ax[SubPlot].plot([ThresValue[SubPlot][0],ThresValue[SubPlot][0]], [min(Ys), max(Ys)], "k--", linewidth=ErrorBar['ErrWid']/2)
                # Outliers
                if Outliers:
                    ax[SubPlot].plot(Xs[Outliers[SubPlot][Scatter]], Ys[Outliers[SubPlot][Scatter]], ErrorBar['OutlierDotsColor'][Scatter]+'o', markersize=ErrorBar['sizedots'])

            # setting plot parameters
            # Axis:
            ax[SubPlot].spines['right'].set_visible(False)
            ax[SubPlot].spines['top'].set_visible(False)
            ax[SubPlot].spines['left'].set_linewidth(axisWidth)
            ax[SubPlot].spines['bottom'].set_linewidth(axisWidth)
            ax[SubPlot].xaxis.set_ticks_position('bottom')
            ax[SubPlot].yaxis.set_ticks_position('left')
            if axisTouching:
                ax[SubPlot].spines['bottom'].set_position(('axes', -0.05))
                ax[SubPlot].spines['left'].set_position(('axes', -0.05))
            for label in (ax[SubPlot].get_xticklabels() + ax[SubPlot].get_yticklabels()):
                label.set_fontname(axisFont['fontname'])
                label.set_fontsize(axisFont['size'])

            # Axis limits and ticks:
            if axeslimitX:
                ax[SubPlot].xaxis.set_ticks(np.arange(axeslimitX[0][SubPlot], axeslimitX[1][SubPlot], axeslimitX[2][SubPlot]))
                ax[SubPlot].set_xlim([axeslimitX[0][SubPlot]+axeslimitX[0][SubPlot]*0.01, axeslimitX[1][SubPlot]+axeslimitX[1][SubPlot]*0.01])
            if axeslimitY:
                ax[SubPlot].yaxis.set_ticks(np.arange(axeslimitY[0][SubPlot], axeslimitY[1][SubPlot], axeslimitY[2][SubPlot]))
                ax[SubPlot].set_ylim([axeslimitY[0][SubPlot]+axeslimitY[0][SubPlot]*0.01, axeslimitY[1][SubPlot]+axeslimitY[1][SubPlot]*0.01])
            ax[SubPlot].xaxis.set_tick_params(width=Ticks['width'], length=Ticks['length'])
            ax[SubPlot].yaxis.set_tick_params(width=Ticks['width'], length=Ticks['length'])


            # Axis Tick labels:
            if SameAxisTicks:
                if AxisTicks[0]:
                    ax[SubPlot].xaxis.set_ticklabels(AxisTicks[0])
                if AxisTicks[1]:
                    ax[SubPlot].yaxis.set_ticklabels(AxisTicks[1])
            else:
                if AxisTicks[0]:
                    ax[SubPlot].xaxis.set_ticklabels(AxisTicks[0][SubPlot])
                if AxisTicks[1]:
                    ax[SubPlot].yaxis.set_ticklabels(AxisTicks[1][SubPlot])

            # Axis lables:
            # X:
            if SameAxisLabel:
                if Row == range(RowSubplots)[-1]:
                    if len(axisLabels[0])>1:
                        ax[SubPlot].set_xlabel(axisLabels[0][Col], **axisFont)
                    elif len(axisLabels[0])==1:
                        ax[SubPlot].set_xlabel(axisLabels[0][0], **axisFont)
            else:
                ax[SubPlot].set_xlabel(axisLabels[0][SubPlot], **axisFont)
            # Y:
            if SameAxisLabel:
                if SubPlot%(Col+1)==0 or SubPlot==1:
                    if len(axisLabels[1])>1:
                        ax[SubPlot].set_ylabel(axisLabels[1][Row], **axisFont)
                    elif len(axisLabels[1])==1:
                        ax[SubPlot].set_ylabel(axisLabels[1][0], **axisFont)
            else:
                ax[SubPlot].set_ylabel(axisLabels[1][SubPlot], **axisFont)

            # Legend: TODO: finish this
            if ConditionLabels:
                ax[SubPlot].legend(loc=LegendPos['LegendPosition'], fontsize=axisFont['size'])

            # Subplot titles:
            if SubplotTitles:
                if SameSubplotTitles:
                    if Col in range(ColSubplots):
                        ax[SubPlot].set_title(SubplotTitles[Col], **axisFont)
                else:
                    ax[SubPlot].set_title(SubplotTitles[SubPlot], **axisFont)


    # Settings for the whole plot
    plt.suptitle(plotTitle,**titleFont)
    plt.tight_layout(pad=FigureLayout[0], w_pad=FigureLayout[1], h_pad=FigureLayout[2])
    fig.subplots_adjust(top=FigureLayout[3])

    if SaveFigName:
        plt.savefig(SaveFigName)

    plt.show()
