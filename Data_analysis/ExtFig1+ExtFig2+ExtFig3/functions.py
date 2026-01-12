import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from statannotations.Annotator import Annotator
import math
from math import remainder, tau
import scipy

# NOTE Parameters
arenaBorderColor = "#BAD7E9"
lightDarkColorPalette = {"Light": "#FC8D03", "Dark": "#5950E9"}
longShortColorPalette = {"Long search": "#EB4143", "Short search": "#3F92E9"}
arenaHighlightColor = "#525252"

GLOBALFONTSIZE = 10
GLOBALTICKWIDTH = 1.6
GLOBALTICKLENGTH = 6
LEVERREGIONCOLOR = "#FC8D03"

# Used from : https://cduvallet.github.io/posts/2018/03/boxplots-in-python
boxprops = {"edgecolor": "k", "linewidth": 1.5}
lineprops = {"color": "k", "linewidth": 1.5}
histprops = {"linewidth": 1.5}
arrowParams = {"headwidth": 4, "headlength": 6, "headaxislength": 5.5}
leftRightColorPalette = {
    "Left": "#EB4143",
    "Right": "#3F92E9",
}

boxplot_kwargs = dict(
    {
        "boxprops": boxprops,
        "medianprops": lineprops,
        "whiskerprops": lineprops,
        "capprops": lineprops,
    }
)
stripplot_kwargs = dict(
    {"linewidth": 1, "size": 5, "alpha": 0.8},
)


# These are helper functions for getting the pairs of elements used for doing stats
import itertools


def get_list_of_pairs_condition_first(elementList, conditionList):
    holder = []
    for condition in conditionList:
        for first, second in itertools.combinations(elementList, 2):
            holder.append(((condition, first), (condition, second)))
    return holder


def get_list_of_pairs_element_first(elementList, conditionList):
    holder = []
    for element in elementList:
        for first, second in itertools.combinations(conditionList, 2):
            holder.append(((first, element), (second, element)))
    return holder


# NOTE Functions
def remove_all_ticks(ax):
    """
    Remove all ticks from the given axis.

    Parameters:
        ax (matplotlib.axis.Axis): The axis object from which ticks will be removed.

    Returns:
        None
    """
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])


def remove_spines(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)


def calculatePvalue(Pvalue):
    """
    Calculate the symbol to plot based on the given p-value.

    Parameters:
        Pvalue (float): The p-value to calculate the symbol for.

    Returns:
        str: The symbol to plot based on the p-value.
    """

    if Pvalue < 0.0001:
        symbolToPlot = "****"
    elif Pvalue < 0.001:
        symbolToPlot = "***"
    elif Pvalue < 0.01:
        symbolToPlot = "**"
    elif Pvalue < 0.05:
        symbolToPlot = "*"
    else:
        symbolToPlot = "ns"

    return symbolToPlot


def draw_stats_bar(ax, A, B, height=0.95, pValue=1):
    """
    Draws a statistics bar on the given axes.

    Parameters:
        ax (Axes): The axes on which to draw the statistics bar.
        A (float): The starting x-coordinate of the bar.
        B (float): The ending x-coordinate of the bar.
        height (float, optional): The height of the bar. Defaults to 0.95.
        pValue (int, optional): The p-value to calculate. Defaults to 1.

    Returns:
        None
    """
    line_x = [A, B]
    line_y = [height, height]

    # Add the horizontal line
    ax.plot(line_x, line_y, color="black", linewidth=1.5, transform=ax.transAxes)

    # Add the two wiskers
    wiskerLeftX = [A, A]
    wiskerLeftY = [height - 0.025, height]

    ax.plot(
        wiskerLeftX, wiskerLeftY, color="black", linewidth=1.5, transform=ax.transAxes
    )

    wiskerRightX = [B, B]
    wiskerRightY = [height - 0.025, height]

    ax.plot(
        wiskerRightX, wiskerRightY, color="black", linewidth=1.5, transform=ax.transAxes
    )

    # Add P-Value
    ax.text(
        (A + B) / 2,
        height + 0.05,
        f"{calculatePvalue(pValue)}",
        fontsize=GLOBALFONTSIZE + 1,
        transform=ax.transAxes,
        ha="center",
        va="center",
    )


def draw_circle(ax, r=44, c=arenaBorderColor, center=(0, 0), lw=2, ls="solid"):

    # Define the center coordinates and radius of the circle
    center = center
    radius = r

    # Create the circle patch
    circle = plt.Circle(
        center, radius, edgecolor=c, facecolor="none", lw=lw, linestyle=ls, alpha=1
    )

    # Add the circle patch to the axes
    ax.add_patch(circle)

    # Set the aspect ratio to 'equal' to ensure the circle appears as a circle
    ax.set_aspect("equal")

    # Set the x and y axis limits
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)


def add_text(ax, x, y, label, fw="normal"):
    ax.text(
        x,
        y,
        label,
        style="normal",
        fontweight=fw,
        fontsize=GLOBALFONTSIZE + 4,
        verticalalignment="center",
        horizontalalignment="center",
        transform=ax.transAxes,
    )


def showImage(ax, imagePath):
    image = plt.imread(imagePath)
    # Get the dimensions of the image
    image_height, image_width, _ = image.shape

    scaled_width = image_width
    scaled_height = image_height

    # Calculate the center position for the image
    center_x = scaled_width / 2
    center_y = scaled_height / 2

    # Set the extent of the image in data coordinates
    extent = [
        center_x - image_width / 2,
        center_x + image_width / 2,
        center_y - image_height / 2,
        center_y + image_height / 2,
    ]

    # Display the image
    ax.imshow(image, extent=extent)

    # Set the aspect ratio to 'equal' to prevent image distortion
    ax.set_aspect("equal")

    # Remove ticks and labels
    ax.axis("off")


def plot_lightDark_histplot(
    ax,
    res,
    xValue="homingErrorAtPeripheryLeverAbsolute",
    xLabel="Error at Peri (Deg.)",
    title="",
    ylabel="Trials",
):

    palette = lightDarkColorPalette

    inputDf = res.copy()
    inputDf["homingErrorAtPeripheryLeverAbsolute"] = np.radians(
        inputDf["homingErrorAtPeripheryLeverAbsolute"]
    )
    inputDf["homingErrorAtPeripheryLeverAbsoluteDegrees"] = np.radians(
        inputDf["homingErrorAtPeripheryLeverAbsoluteDegrees"]
    )
    inputDf["headingError"] = np.radians(inputDf["headingError"])
    inputDf["light"] = inputDf["light"].replace({"light": "Light", "dark": "Dark"})

    b = sns.histplot(
        data=inputDf, x=xValue, hue="light", bins=40, kde=True, palette=palette
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_title(f"{title}", fontsize=GLOBALFONTSIZE)
    ax.set_ylabel(ylabel, fontsize=GLOBALFONTSIZE)
    ax.set_xlabel(xLabel, fontsize=GLOBALFONTSIZE)
    ax.tick_params(
        axis="both",
        which="both",
        labelsize=GLOBALFONTSIZE,
        width=GLOBALTICKWIDTH,
        length=GLOBALTICKLENGTH,
    )

    ax.set_xticks(ticks=[0, np.pi / 2, np.pi])
    ax.set_xticklabels(["0", "$\pi$/2", "$\pi$"])

    sns.move_legend(
        b,
        "lower center",
        bbox_to_anchor=(0.7, 0.6),
        ncol=1,
        title=None,
        frameon=False,
        fontsize=GLOBALFONTSIZE,
    )


def plot_lightDark_boxplot(
    ax,
    res,
    yValue="homingErrorAtPeripheryLeverAbsolute",
    xLabel="Homing Movement Angle (Deg.)",
    title="",
    ylabel="Trials",
):

    inputDf = res.copy()
    inputDf["homingErrorAtPeripheryLeverAbsolute"] = np.radians(
        inputDf["homingErrorAtPeripheryLeverAbsolute"]
    )
    inputDf["homingErrorAtPeripheryLeverAbsoluteDegrees"] = np.radians(
        inputDf["homingErrorAtPeripheryLeverAbsoluteDegrees"]
    )
    inputDf["headingError"] = np.radians(inputDf["headingError"])
    inputDf["light"] = inputDf["light"].replace({"light": "Light", "dark": "Dark"})

    palette = {"Light": "#FC8D03", "Dark": "#5950E9"}

    palette = lightDarkColorPalette

    order = ["Light", "Dark"]

    error_name = yValue

    def mySummary(df):
        error = np.nanmedian(df[f"{error_name}"])
        return pd.DataFrame({"error": [error]})

    groupedRes = inputDf.groupby(["light", "subject"]).apply(mySummary).reset_index()

    b = sns.boxplot(
        data=groupedRes,
        y="error",
        x="light",
        palette=palette,
        showfliers=False,
        order=order,
        **boxplot_kwargs,
    )
    sns.stripplot(
        data=groupedRes,
        y="error",
        x="light",
        palette=palette,
        order=order,
        edgecolor="black",
        **stripplot_kwargs,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_title(f"{title}", y=1.1, fontsize=GLOBALFONTSIZE, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=GLOBALFONTSIZE)
    ax.set_xlabel(xLabel, fontsize=GLOBALFONTSIZE)

    # ax.set_ylim(0,3.15)
    ax.set_yticks(ticks=[0, np.pi / 4, np.pi / 2])
    ax.set_yticklabels(["0", "$\pi$/4", "$\pi$/2"])

    ax.tick_params(
        axis="both",
        which="both",
        labelsize=GLOBALFONTSIZE,
        width=GLOBALTICKWIDTH,
        length=GLOBALTICKLENGTH,
    )

    pairs = [[("Light"), ("Dark")]]
    annotator = Annotator(b, pairs, data=groupedRes, y="error", x="light", order=order)
    annotator.configure(test="Wilcoxon", text_format="star", loc="inside")
    annotator.apply_and_annotate()

    number = len(groupedRes.subject.unique())
    ax.text(
        0.07, 0.93, f"N = {number}", transform=ax.transAxes, fontsize=GLOBALFONTSIZE
    )


def plot_longShort_histplot(
    ax,
    res,
    lightCondition="light",
    xValue="homingErrorAtPeripheryLeverAbsolute",
    xLabel="Heading Error (Deg.)",
    title="",
    ylabel="Trials",
):
    """
    Plots a histogram of the given data with respect to the specified parameters.

    Parameters:
        ax (matplotlib.axes.Axes): The axes object to plot the histogram on.
        res (pandas.DataFrame): The input data.
        lightCondition (str, optional): The light condition to filter the data on. Defaults to 'light'.
        xValue (str, optional): The column name of the data to plot on the x-axis. Defaults to 'homingErrorAtPeripheryLeverAbsolute'.
        xLabel (str, optional): The label for the x-axis. Defaults to 'Heading Error (Deg.)'.
        title (str, optional): The title for the plot. Defaults to an empty string.
        ylabel (str, optional): The label for the y-axis. Defaults to 'Trials'.

    Returns:
        None
    """

    inputDf = res.copy()
    inputDf["homingErrorAtPeripheryLeverAbsolute"] = np.radians(
        inputDf["homingErrorAtPeripheryLeverAbsolute"]
    )
    inputDf["homingErrorAtPeripheryLeverAbsoluteDegrees"] = np.radians(
        inputDf["homingErrorAtPeripheryLeverAbsoluteDegrees"]
    )
    inputDf["headingError"] = np.radians(inputDf["headingError"])
    inputDf["searchPathSorL"] = inputDf["searchPathSorL"].replace(
        {"shortPath": "Short search", "longPath": "Long search"}
    )

    palette = longShortColorPalette

    hue_order = ["Short search", "Long search"]
    b = sns.histplot(
        data=inputDf[(inputDf.light == lightCondition)],
        x=xValue,
        hue="searchPathSorL",
        bins=40,
        kde=True,
        palette=palette,
        hue_order=hue_order,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_title(f"{title}", fontsize=GLOBALFONTSIZE)
    ax.set_ylabel(ylabel, fontsize=GLOBALFONTSIZE)
    ax.set_xlabel(xLabel, fontsize=GLOBALFONTSIZE)
    ax.tick_params(
        axis="both",
        which="both",
        labelsize=GLOBALFONTSIZE,
        width=GLOBALTICKWIDTH,
        length=GLOBALTICKLENGTH,
    )

    ax.set_xticks(ticks=[0, np.pi / 2, np.pi])
    ax.set_xticklabels(["0", "$\pi$/2", "$\pi$"])

    sns.move_legend(
        b,
        "lower center",
        bbox_to_anchor=(0.47, 0.93),
        ncol=1,
        title=None,
        frameon=False,
        fontsize=GLOBALFONTSIZE,
    )


def plot_longShort_boxplot(
    ax,
    res,
    yValue="homingErrorAtPeripheryLeverAbsolute",
    xLabel="Homing Movement Angle (Deg.)",
    title="",
    ylabel="Trials",
):

    inputDf = res.copy()
    inputDf["homingErrorAtPeripheryLeverAbsolute"] = np.radians(
        inputDf["homingErrorAtPeripheryLeverAbsolute"]
    )
    inputDf["homingErrorAtPeripheryLeverAbsoluteDegrees"] = np.radians(
        inputDf["homingErrorAtPeripheryLeverAbsoluteDegrees"]
    )
    inputDf["headingError"] = np.radians(inputDf["headingError"])
    inputDf["light"] = inputDf["light"].replace({"light": "Light", "dark": "Dark"})
    inputDf["searchPathSorL"] = inputDf["searchPathSorL"].replace(
        {"shortPath": "Short search", "longPath": "Long search"}
    )

    palette = {"Short search": "#FF5630", "Long search": "#3F92E9"}
    palette = longShortColorPalette

    order = ["Light", "Dark"]
    hue_order = ["Short search", "Long search"]

    error_name = yValue

    def mySummary(df):
        error = np.nanmedian(df[f"{error_name}"])
        return pd.DataFrame({"error": [error]})

    groupedRes = (
        inputDf.groupby(["light", "searchPathSorL", "subject"])
        .apply(mySummary)
        .reset_index()
    )

    b = sns.boxplot(
        data=groupedRes,
        y="error",
        x="light",
        hue="searchPathSorL",
        palette=palette,
        showfliers=False,
        order=order,
        hue_order=hue_order,
        **boxplot_kwargs,
    )
    sns.stripplot(
        data=groupedRes,
        y="error",
        x="light",
        hue="searchPathSorL",
        palette=palette,
        order=order,
        edgecolor="black",
        dodge=True,
        hue_order=hue_order,
        legend=False,
        **stripplot_kwargs,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_title(f"{title}", fontsize=GLOBALFONTSIZE)
    ax.set_ylabel(ylabel, fontsize=GLOBALFONTSIZE)
    ax.set_xlabel(xLabel, fontsize=GLOBALFONTSIZE)

    ax.set_yticks(ticks=[0, np.pi / 4, np.pi / 2])
    ax.set_yticklabels(["0", "$\pi$/4", "$\pi$/2"])
    # ax.set_ylim(0,np.pi/2)

    ax.tick_params(
        axis="both",
        which="both",
        labelsize=GLOBALFONTSIZE,
        width=GLOBALTICKWIDTH,
        length=GLOBALTICKLENGTH,
    )

    sns.move_legend(
        b,
        "lower center",
        bbox_to_anchor=(0.47, 0.93),
        ncol=1,
        title=None,
        frameon=False,
        fontsize=GLOBALFONTSIZE,
    )

    s = groupedRes.light.unique()
    holder = groupedRes.searchPathSorL.unique()

    pairs = get_list_of_pairs_condition_first(holder, s)

    annotator = Annotator(
        b,
        pairs,
        data=groupedRes,
        y="error",
        x="light",
        order=order,
        hue="searchPathSorL",
        hue_order=hue_order,
    )
    annotator.configure(test="Wilcoxon", text_format="star", loc="inside")
    annotator.apply_and_annotate()

    number = len(groupedRes.subject.unique())
    ax.text(
        0.07, 0.88, f"N = {number}", transform=ax.transAxes, fontsize=GLOBALFONTSIZE
    )


def scatter_lever_position_all_points(ax, inputDf):
    b = sns.scatterplot(
        data=inputDf,
        x="leverX",
        y="leverY",
        facecolor="#FFFFFF",
        edgecolor="#DFD7BF",
        alpha=1,
        s=16,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)


def draw_rectangle(ax, width=40, height=40, c="black", center=(0, 0), lw=4):

    centerR = (center[0] - width / 2, center[1] - height / 2)
    # Create the circle patch
    rectangle = plt.Rectangle(
        centerR,
        width,
        height,
        edgecolor=c,
        facecolor="none",
        lw=lw,
        linestyle="solid",
        alpha=0.8,
    )

    # Add the circle patch to the axes
    ax.add_patch(rectangle)

    # Set the aspect ratio to 'equal' to ensure the circle appears as a circle
    ax.set_aspect("equal")

    # Set the x and y axis limits
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)


def scatter_lever_position(ax, inputDf, lc):
    b = sns.scatterplot(data=inputDf, x="leverX", y="leverY", color=LEVERREGIONCOLOR)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.set_title(f"{lc}", transform=ax.transAxes, fontsize=GLOBALFONTSIZE)
    ax.set_xlabel("X coordinate (cm)", fontsize=GLOBALFONTSIZE)
    ax.set_ylabel("Y coordinate (cm)", fontsize=GLOBALFONTSIZE)

    custom_ticks = [-40, 0, 40]
    ax.set_xticks(custom_ticks)
    ax.set_yticks(custom_ticks)
    ax.tick_params(
        axis="both",
        which="both",
        labelsize=GLOBALFONTSIZE,
        width=GLOBALTICKWIDTH,
        length=GLOBALTICKLENGTH,
    )


def plot_homingPath(ax, inputDf, instanDf, legend=True, lc="Dark"):

    color_dict = {"Left": "#525FE1", "Right": "#F86F03"}
    color_dict = leftRightColorPalette

    mergedPlot = inputDf.merge(
        instanDf,
        left_on=["sessionName", "trialNo"],
        right_on=["sessionName", "trialNo"],
        how="left",
    )
    mergedPlot = mergedPlot.sort_values("homingLeftRight")

    # Replace left -> Left, right -> Right
    mergedPlot["homingLeftRight"] = mergedPlot["homingLeftRight"].replace(
        {"left": "Left", "right": "Right"}
    )

    mergedPlot = mergedPlot[(mergedPlot.x.abs() < 37.5) & (mergedPlot.y.abs() < 37.5)]

    b = sns.scatterplot(
        data=mergedPlot[mergedPlot.condition == "homingFromLeavingLeverToPeriphery"],
        x="x",
        y="y",
        hue="homingLeftRight",
        legend=legend,
        palette=color_dict,
        s=12,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.set_xlabel("X coordinate (cm)", fontsize=GLOBALFONTSIZE)
    ax.set_ylabel("Y coordinate (cm)", fontsize=GLOBALFONTSIZE)
    ax.set_title(
        f"{lc}",
        y=1.17,
        fontweight="bold",
        transform=ax.transAxes,
        fontsize=GLOBALFONTSIZE,
    )

    # Set xlim and ylim, as well as the ticks
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)

    custom_ticks = [-40, 0, 40]
    ax.set_xticks(custom_ticks)
    ax.set_yticks(custom_ticks)
    ax.tick_params(
        axis="both",
        which="both",
        labelsize=GLOBALFONTSIZE,
        width=GLOBALTICKWIDTH,
        length=GLOBALTICKLENGTH,
    )

    if legend:
        sns.move_legend(
            b,
            "lower center",
            bbox_to_anchor=(0.5, 0.92),
            ncol=2,
            title=None,
            frameon=False,
            fontsize=GLOBALFONTSIZE,
        )


# Make Left Right Plot
def makePlots(inputDf, gs, fig, instanDf, row=0, mName="jp"):

    # Data Used For Making the Plot
    sns.set_theme(style="ticks")
    resS = inputDf[inputDf.light == "light"]
    resP = resS[
        (
            np.sqrt(
                (resS.startPositionHoming_x + 20) ** 2
                + (resS.startPositionHoming_y) ** 2
            )
            < 5
        )
        | (
            np.sqrt(
                (resS.startPositionHoming_x - 20) ** 2
                + (resS.startPositionHoming_y) ** 2
            )
            < 5
        )
    ]

    resP = resP.dropna(subset="startPositionHoming_x")
    ###################################

    # Add the homing Paths
    ax0 = fig.add_subplot(gs[row, 0])  #
    plot_homingPath(ax0, resP, instanDf, lc=f"Light")

    draw_circle(ax0, r=5, c=arenaHighlightColor, center=(-20, 0), lw=2)
    draw_circle(ax0, r=5, c=arenaHighlightColor, center=(20, 0), lw=2)
    draw_circle(ax0)
    draw_circle(ax0, r=40, lw=1, ls="dashed")

    ##########SECOND############

    resS = inputDf[inputDf.light == "dark"]
    resP = resS[
        (
            np.sqrt(
                (resS.startPositionHoming_x + 20) ** 2
                + (resS.startPositionHoming_y) ** 2
            )
            < 5
        )
        | (
            np.sqrt(
                (resS.startPositionHoming_x - 20) ** 2
                + (resS.startPositionHoming_y) ** 2
            )
            < 5
        )
    ]

    resP = resP.dropna(subset="startPositionHoming_x")
    ###################################

    # Add the homing Paths
    ax0 = fig.add_subplot(gs[row, 1])  #
    plot_homingPath(ax0, resP, instanDf, legend=False, lc=f"Dark")

    draw_circle(ax0, r=5, c=arenaHighlightColor, center=(-20, 0), lw=2)
    draw_circle(ax0, r=5, c=arenaHighlightColor, center=(20, 0), lw=2)
    draw_circle(ax0)
    draw_circle(ax0, r=40, lw=1, ls="dashed")


def scatter_homing_start_position_all_points(ax, inputDf):
    b = sns.scatterplot(
        data=inputDf,
        x="startPositionHoming_x",
        y="startPositionHoming_y",
        color="#F7F7F7",
        edgecolor="#525252",
        alpha=0.05,
        s=12,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)


def scatter_homing_start_position(ax, inputDf, lc):

    color_dict = {"Left": "#525FE1", "Right": "#F86F03"}
    color_dict = leftRightColorPalette

    inputDf = inputDf.copy()
    inputDf["homingLeftRight"] = inputDf["homingLeftRight"].replace(
        {"left": "Left", "right": "Right"}
    )

    b = sns.scatterplot(
        data=inputDf,
        x="startPositionHoming_x",
        y="startPositionHoming_y",
        hue="homingLeftRight",
        palette=color_dict,
        s=12,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.set_title(f"{lc}", y=0.97, transform=ax.transAxes, fontsize=GLOBALFONTSIZE)
    ax.set_xlabel("X coordinate (cm)", fontsize=GLOBALFONTSIZE)
    ax.set_ylabel("Y coordinate (cm)", fontsize=GLOBALFONTSIZE)

    # Set xlim and ylim, as well as the ticks
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)

    custom_ticks = [-40, 0, 40]
    ax.set_xticks(custom_ticks)
    ax.set_yticks(custom_ticks)
    ax.tick_params(
        axis="both",
        which="both",
        labelsize=GLOBALFONTSIZE,
        width=GLOBALTICKWIDTH,
        length=GLOBALTICKLENGTH,
    )

    sns.move_legend(
        b,
        "lower center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=2,
        title=None,
        frameon=False,
        fontsize=GLOBALFONTSIZE,
    )


def plot_leftRight_histplot(
    ax,
    res,
    lightCondition="light",
    xValue="homingErrorAtPeripheryLeverAbsolute",
    xLabel="Homing heading dir. (rad)",
    title="",
    ylabel="Trials",
    legend=True,
):
    """
    Plot a histogram of the homing error at the periphery lever for left and right homing directions.

    Parameters:
        - ax: The matplotlib axes object to plot the histogram on.
        - res: The input DataFrame containing the data to be plotted.
        - lightCondition: The light condition to filter the data by (default: 'light').
        - xValue: The column name in the DataFrame to use as the x-axis values (default: 'homingErrorAtPeripheryLeverAbsolute').
        - xLabel: The label for the x-axis (default: 'Homing heading dir. (rad)').
        - title: The title for the plot (default: '').
        - ylabel: The label for the y-axis (default: 'Trials').
        - legend: Whether to show a legend on the plot (default: True).

    Returns:
        None
    """

    palette = {"Left": "#525FE1", "Right": "#F86F03"}
    palette = leftRightColorPalette

    inputDf = res.copy()
    inputDf["medianMVDeviationRoomReference"] = np.radians(
        inputDf["medianMVDeviationRoomReference"]
    )
    inputDf["medianMVDeviationRoomReference"] = inputDf[
        "medianMVDeviationRoomReference"
    ].apply(lambda x: math.remainder(x, tau))
    inputDf["homingLeftRight"] = inputDf["homingLeftRight"].replace(
        {"left": "Left", "right": "Right"}
    )

    mask = (inputDf["medianMVDeviationRoomReference"] >= -np.pi / 2) & (
        inputDf["medianMVDeviationRoomReference"] <= np.pi / 2
    )

    inputDf = inputDf[mask]

    b = sns.histplot(
        data=inputDf[(inputDf.light == lightCondition)],
        x=xValue,
        hue="homingLeftRight",
        bins=40,
        kde=True,
        palette=palette,
        legend=legend,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_title(
        f"{title}",
        y=1.2,
        fontweight="bold",
        transform=ax.transAxes,
        fontsize=GLOBALFONTSIZE,
    )
    ax.set_ylabel(ylabel, fontsize=GLOBALFONTSIZE)
    ax.set_xlabel(xLabel, fontsize=GLOBALFONTSIZE)

    ax.set_xticks(ticks=[-np.pi / 2, 0, np.pi / 2])
    ax.set_xticklabels([r"-$\pi$/2", "0", "$\pi$/2"])

    ax.tick_params(
        axis="both",
        which="both",
        labelsize=GLOBALFONTSIZE,
        width=GLOBALTICKWIDTH,
        length=GLOBALTICKLENGTH,
    )

    if legend:
        sns.move_legend(
            b,
            "lower center",
            bbox_to_anchor=(0.5, 0.95),
            ncol=2,
            title=None,
            frameon=False,
            fontsize=GLOBALFONTSIZE,
        )


def plot_leftRight_boxplot(
    ax,
    res,
    yValue="medianMVDeviationRoomReference",
    xLabel="Homing heading dir. (rad)",
    title="",
    ylabel="Trials",
):

    palette = {"Left": "#525FE1", "Right": "#F86F03"}

    palette = leftRightColorPalette

    order = ["light", "dark"]
    hue_order = ["Left", "Right"]

    inputDf = res.copy()
    inputDf["medianMVDeviationRoomReference"] = np.radians(
        inputDf["medianMVDeviationRoomReference"]
    )
    inputDf["medianMVDeviationRoomReference"] = inputDf[
        "medianMVDeviationRoomReference"
    ].apply(lambda x: math.remainder(x, tau))
    inputDf["homingLeftRight"] = inputDf["homingLeftRight"].replace(
        {"left": "Left", "right": "Right"}
    )

    error_name = yValue

    def mySummary(df):
        error = np.nanmedian(df[f"{error_name}"])
        return pd.DataFrame({"error": [error]})

    groupedRes = (
        inputDf.groupby(["light", "homingLeftRight", "subject"])
        .apply(mySummary)
        .reset_index()
    )

    b = sns.boxplot(
        data=groupedRes,
        y="error",
        x="light",
        hue="homingLeftRight",
        palette=palette,
        showfliers=False,
        order=order,
        hue_order=hue_order,
        **boxplot_kwargs,
    )
    sns.stripplot(
        data=groupedRes,
        y="error",
        x="light",
        hue="homingLeftRight",
        palette=palette,
        order=order,
        edgecolor="black",
        dodge=True,
        legend=False,
        hue_order=hue_order,
        **stripplot_kwargs,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_title(f"{title}", fontsize=GLOBALFONTSIZE)
    ax.set_ylabel(ylabel, fontsize=GLOBALFONTSIZE)
    ax.set_xlabel(xLabel, fontsize=GLOBALFONTSIZE)

    ax.set_yticks(ticks=[-np.pi / 3, 0, np.pi / 3])
    ax.set_yticklabels([r"-$\pi$/3", "0", "$\pi$/3"])

    sns.move_legend(
        b,
        "lower center",
        bbox_to_anchor=(0.5, 1),
        ncol=2,
        title=None,
        frameon=False,
        fontsize=GLOBALFONTSIZE,
    )

    s = groupedRes.light.unique()
    ax.set_xticklabels(["Light", "Dark"])
    ax.tick_params(
        axis="both",
        which="both",
        labelsize=GLOBALFONTSIZE,
        width=GLOBALTICKWIDTH,
        length=GLOBALTICKLENGTH,
    )

    pairs = get_list_of_pairs_condition_first(hue_order, s)

    annotator = Annotator(
        b,
        pairs,
        data=groupedRes,
        y="error",
        x="light",
        order=order,
        hue="homingLeftRight",
        hue_order=hue_order,
    )
    annotator.configure(test="Wilcoxon", text_format="star", loc="inside")
    annotator.apply_and_annotate()

    number = len(groupedRes.subject.unique())
    ax.text(
        0.07, 0.93, f"N = {number}", transform=ax.transAxes, fontsize=GLOBALFONTSIZE
    )


def plot_leftRight_difference_histplot(
    ax,
    res,
    xValue="homingErrorAtPeripheryLeverAbsolute",
    xLabel="Homing heading diff. (rad)",
    title="",
    ylabel="Trials",
):

    palette = {"Light": "#525FE1", "Dark": "#F86F03"}

    palette = lightDarkColorPalette

    inputDf = res.copy()
    inputDf["medianMVDeviationRoomReference"] = np.radians(
        inputDf["medianMVDeviationRoomReference"]
    )
    inputDf["medianMVDeviationRoomReference"] = inputDf[
        "medianMVDeviationRoomReference"
    ].apply(lambda x: math.remainder(x, tau))
    inputDf["homingLeftRight"] = inputDf["homingLeftRight"].replace(
        {"left": "Left", "right": "Right"}
    )
    inputDf["light"] = inputDf["light"].replace({"light": "Light", "dark": "Dark"})

    mask = (inputDf["medianMVDeviationRoomReference"] >= -np.pi / 2) & (
        inputDf["medianMVDeviationRoomReference"] <= np.pi / 2
    )

    inputDf = inputDf[mask]

    yValue = "medianMVDeviationRoomReference"
    error_name = yValue

    def mySummary(df):
        error = np.nanmedian(df[f"{error_name}"])
        return pd.DataFrame({"error": [error]})

    groupedRes = (
        inputDf.groupby(["light", "homingLeftRight", "subject"])
        .apply(mySummary)
        .reset_index()
    )
    diffDf = groupedRes.pivot_table(
        values="error", columns=["homingLeftRight"], index=["subject", "light"]
    ).reset_index()
    diffDf["leftRightDiff"] = diffDf["Left"] - diffDf["Right"]

    hue_order = ["Light", "Dark"]
    b = sns.histplot(
        data=diffDf,
        x="leftRightDiff",
        hue="light",
        bins=5,
        kde=True,
        palette=palette,
        hue_order=hue_order,
    )

    statsRes = scipy.stats.wilcoxon(
        diffDf[diffDf.light == "Light"]["leftRightDiff"],
        diffDf[diffDf.light == "Dark"]["leftRightDiff"],
    )
    print(statsRes)
    draw_stats_bar(ax, A=0.35, B=0.65, height=0.93, pValue=statsRes.pvalue)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_title(f"{title}", fontsize=GLOBALFONTSIZE)
    ax.set_ylabel(ylabel, fontsize=GLOBALFONTSIZE)
    ax.set_xlabel(xLabel, fontsize=GLOBALFONTSIZE)
    ax.set_ylim(0, 20)

    ax.set_xticks(ticks=[-np.pi / 6, 0, np.pi / 6, np.pi / 3])
    ax.set_xticklabels([r"-$\pi$/6", "0", "$\pi$/6", "$\pi$/3"])

    ax.tick_params(
        axis="both",
        which="both",
        labelsize=GLOBALFONTSIZE,
        width=GLOBALTICKWIDTH,
        length=GLOBALTICKLENGTH,
    )
    sns.move_legend(
        b,
        "lower center",
        bbox_to_anchor=(0.5, 1),
        ncol=2,
        title=None,
        frameon=False,
        fontsize=GLOBALFONTSIZE,
    )


def plot_arrow_map(ax, inputDf, bins=7, lc="Light"):

    x = inputDf["startPositionHoming_x"]
    y = inputDf["startPositionHoming_y"]
    score = inputDf["medianMVDeviationRoomReference"]

    num_bins = bins

    # Create a 2D histogram
    hist, xedges, yedges = np.histogram2d(x, y, bins=num_bins)

    # Calculate the average score for each bin
    average_score = np.zeros_like(hist)
    for i in range(num_bins):
        for j in range(num_bins):
            mask = (
                (x >= xedges[i])
                & (x < xedges[i + 1])
                & (y >= yedges[j])
                & (y < yedges[j + 1])
            )
            average_score[i, j] = np.median(score[mask])

    # Create a meshgrid for quiver plot
    x_quiver, y_quiver = np.meshgrid(
        xedges[:-1] + (xedges[1] - xedges[0]) / 2,
        yedges[:-1] + (yedges[1] - yedges[0]) / 2,
    )

    # Calculate the angles and magnitudes for the arrows
    angles = np.radians(average_score)
    magnitudes = np.ones_like(average_score)

    u = magnitudes * np.cos(angles)
    v = magnitudes * np.sin(angles)

    # Plot the quiver plot
    ax.quiver(x_quiver, y_quiver, u.T, v.T, pivot="mid", scale=13, **arrowParams)

    # Add labels and title
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.set_xlabel("X coordinate (cm)", fontsize=GLOBALFONTSIZE)
    ax.set_ylabel("Y coordinate (cm)", fontsize=GLOBALFONTSIZE)
    ax.set_title(f"{lc}", fontsize=GLOBALFONTSIZE, fontweight="bold")

    custom_ticks = [-40, 0, 40]
    ax.set_xticks(custom_ticks)
    ax.set_yticks(custom_ticks)
    ax.tick_params(
        axis="both",
        which="both",
        labelsize=GLOBALFONTSIZE,
        width=GLOBALTICKWIDTH,
        length=GLOBALTICKLENGTH,
    )
