def ErrorBarsForMeans(data,
                      SpreadOfX = 0.1,
                      YError = 'CI',
                      plotsize = [15,15],
                      axeslimit = [],
                      axisLabels = [],
                      SameAxisLabel = True,
                      SubplotTitles = [],
                      SameSubplotTitles = True,
                      plotTitle = [],
                      ThresValue = [],
                      AxisTicks = [],
                      SameAxisTicks = True,
                      SaveFigName = [],
                      Outliers = [],
                      showplot = True,
                      Ticks = {'width':2, 'length':6},
                      axisTouching = False,
                      Axis = 'left',
                      titleFont = {'fontname':'Arial', 'size':'28', 'color':'black', 'weight':'normal','verticalalignment':'bottom'},
                      axisFont = {'fontname':'Arial', 'size':'24'},
                      axisWidth = 2,
                      ErrorBar = {'ErrLen':0.15, 'ErrWid1':6, 'ErrWid2':6, 'sizeMean':24, 'sizedots':10, 'ErrColor':'k', 'DotsColor':'y','DotsTransp':0.9,'OutlierDotsColor':'r'},
                      FigureLayout = [1,1,1,0.95]):

    # imports
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats as sc

    # parameters for the figure:
    ColSubplots = len(data)
    RowSubplots = len(data[0])
    locx = range(1,len(data[0][0])+1)
    NSubPlots = ColSubplots*RowSubplots

    # Figure
    fig, ax = plt.subplots(RowSubplots,ColSubplots, figsize=(plotsize[0], plotsize[1]), squeeze=False)
    
    ax = ax.ravel()

    SubPlot = -1
    for Row in range(RowSubplots):
        for Col in range(ColSubplots):

            SubPlot = SubPlot+1

            for Bars in range(len(data[Col][Row])):

                # data:
                Ys = data[Col][Row][Bars]
                Xs = np.random.normal(loc=locx[Bars], scale=SpreadOfX, size=len(Ys))
                # 95% Conf.Intervals
                if YError == 'CI':
                    LowerError = sc.t.interval(0.95, len(Ys)-1, loc=np.mean(Ys), scale=sc.sem(Ys))[0]
                    UpperError = sc.t.interval(0.95, len(Ys)-1, loc=np.mean(Ys), scale=sc.sem(Ys))[1]
                # standard deviation
                elif YError == 'STD':
                    LowerError = np.mean(Ys)-np.std(Ys)
                    UpperError = np.mean(Ys)+np.std(Ys)
                # SEM
                elif YError == 'SEM':
                    LowerError = np.mean(Ys)-sc.sem(Ys)
                    UpperError = np.mean(Ys)+sc.sem(Ys)
                # Points
                ax[SubPlot].plot(Xs, Ys, 'o', color = ErrorBar['DotsColor'][Bars],  markersize=ErrorBar['sizedots'], alpha=ErrorBar['DotsTransp'])
                # Error bar:
                ax[SubPlot].plot([locx[Bars],locx[Bars]],[LowerError,UpperError], ErrorBar['ErrColor'][Bars]+'-', linewidth=ErrorBar['ErrWid1'])
                ax[SubPlot].plot([locx[Bars]-ErrorBar['ErrLen'],locx[Bars]+ErrorBar['ErrLen']],[UpperError,UpperError], ErrorBar['ErrColor'][Bars]+'-', linewidth=ErrorBar['ErrWid2'])
                ax[SubPlot].plot([locx[Bars]-ErrorBar['ErrLen'],locx[Bars]+ErrorBar['ErrLen']],[LowerError,LowerError], ErrorBar['ErrColor'][Bars]+'-', linewidth=ErrorBar['ErrWid2'])
                ax[SubPlot].plot(locx[Bars],np.mean(Ys),"ko", markersize=ErrorBar['sizeMean'])
                
                # Outliers
                if Outliers:
                    ax[SubPlot].plot(Xs[Outliers], Ys[Outliers], ErrorBar['OutlierDotsColor']+'o', markersize=ErrorBar['sizedots'])

            # showing the threshold value
            if ThresValue:
                ax[SubPlot].plot([locx[0]-1,locx[-1]+1], [ThresValue[SubPlot],ThresValue[SubPlot]], "k--", linewidth=ErrorBar['ErrWid2']/2)


            # setting plot parameters
            # Axis:
            ax[SubPlot].spines['top'].set_visible(False)
            ax[SubPlot].spines['bottom'].set_linewidth(axisWidth)
            ax[SubPlot].xaxis.set_ticks_position('bottom')
            if Axis=='right':
                rot = 270
                pd = 31
                ax[SubPlot].spines['left'].set_visible(False)
                ax[SubPlot].spines['right'].set_linewidth(axisWidth)
                ax[SubPlot].yaxis.set_ticks_position('right')
                ax[SubPlot].yaxis.set_label_position('right')
            elif Axis=='left':
                rot = 90
                pd = 7
                ax[SubPlot].spines['right'].set_visible(False)
                ax[SubPlot].spines['left'].set_linewidth(axisWidth)
                ax[SubPlot].yaxis.set_ticks_position('left')
            elif Axis=='both':
                rot = 90
                pd = 7
                ax[SubPlot].spines['right'].set_linewidth(axisWidth)
                ax[SubPlot].spines['left'].set_linewidth(axisWidth)
                ax[SubPlot].yaxis.set_ticks_position('both')
                ax[SubPlot].yaxis.set_label_position('left')
            if axisTouching:
                ax[SubPlot].spines['bottom'].set_position(('axes', -0.05))
                if Axis=='right':
                    ax[SubPlot].spines['right'].set_position(('axes', 1.05))
                elif Axis=='left':
                    ax[SubPlot].spines['left'].set_position(('axes', -0.05))
                elif Axis=='both':
                    ax[SubPlot].spines['right'].set_position(('axes', 1.05))
                    ax[SubPlot].spines['left'].set_position(('axes', -0.05))        
            for label in (ax[SubPlot].get_xticklabels() + ax[SubPlot].get_yticklabels()):
                label.set_fontname(axisFont['fontname'])
                label.set_fontsize(axisFont['size'])

            # Axis limits and ticks:
            if axeslimit:
                ax[SubPlot].yaxis.set_ticks(np.arange(axeslimit[0][SubPlot], axeslimit[1][SubPlot], axeslimit[2][SubPlot]))
                ax[SubPlot].set_ylim([axeslimit[0][SubPlot], axeslimit[1][SubPlot]])
            ax[SubPlot].xaxis.set_ticks(locx)
            ax[SubPlot].set_xlim([locx[0]-1, locx[-1]+1])
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
                        ax[SubPlot].set_ylabel(axisLabels[1][Row], rotation=rot, labelpad=pd, **axisFont)
                    elif len(axisLabels[1])==1:
                        ax[SubPlot].set_ylabel(axisLabels[1][0], rotation=rot, labelpad=pd, **axisFont)
            else:
                ax[SubPlot].set_ylabel(axisLabels[1][SubPlot], rotation=rot, labelpad=pd, **axisFont)

            # Legend: TODO: finish this
            #ax[SubPlot].legend(loc='lower right')

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
        plt.savefig(SaveFigName, bbox_inches='tight')

    if showplot:
        plt.show()
    else:
        return fig,ax
