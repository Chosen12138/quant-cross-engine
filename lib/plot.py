
import os
from datetime import datetime
from gettext import NullTranslations, translation
from conf.path_config import *

__all__ = ['plot_result']


class Localization(object):

    def __init__(self, trans=None):
        self.trans = NullTranslations() if trans is None else trans

    def set_locale(self, locales, trans_dir=None):
        if locales[0] is None or "en" in locales[0].lower():
            self.trans = NullTranslations()
            return
        if "cn" in locales[0].lower():
            locales = ["zh_Hans_CN"]
        try:
            if trans_dir is None:
                trans_dir = os.path.join(
                    os.path.dirname(
                        os.path.abspath(
                            __file__,
                        ),
                    ),
                    "translations"
                )
            self.trans = translation(
                domain="messages",
                localedir=trans_dir,
                languages=locales,
            )
        except Exception as e:
            # system_log.debug(e)
            self.trans = NullTranslations()


localization = Localization()


def gettext(message):
    return localization.trans.gettext(message)


def max_ddd(arr):
    max_seen = arr[0]
    ddd_start, ddd_end = 0, 0
    ddd = 0
    start = 0
    in_draw_down = False

    for i in range(len(arr)):
        if arr[i] > max_seen:
            if in_draw_down:
                in_draw_down = False
                if i - start > ddd:
                    ddd = i - start
                    ddd_start = start
                    ddd_end = i - 1
            max_seen = arr[i]
        elif arr[i] < max_seen:
            if not in_draw_down:
                in_draw_down = True
                start = i - 1

    if arr[i] < max_seen:
        if i - start > ddd:
            return start, i

    return ddd_start, ddd_end


def plot_result(result_dict, show_windows=True, save_path=None):
    from matplotlib import rcParams, gridspec, ticker, image as mpimg, pyplot as plt
    from matplotlib.font_manager import findfont, FontProperties
    import numpy as np

    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = [
        u'Microsoft Yahei',
        u'Heiti SC',
        u'Heiti TC',
        u'STHeiti',
        u'WenQuanYi Zen Hei',
        u'WenQuanYi Micro Hei',
        u"文泉驿微米黑",
        u'SimHei',
    ] + rcParams['font.sans-serif']
    rcParams['axes.unicode_minus'] = False

    use_chinese_fonts = True
    font = findfont(FontProperties(family=['sans-serif']))
    if "/matplotlib/" in font:
        use_chinese_fonts = False

    summary = result_dict["performance_summary"]

    title = summary['strategy_name']

    back_test_result = result_dict["back_test_result"]
    index = back_test_result.index

    portfolio = back_test_result[['Portfolio_Equity']]
    benchmark_portfolio = back_test_result[['Benchmark_Equity']]
    # max drawdown
    portfolio_value = back_test_result.ASSET
    xs = portfolio_value.values
    rt = portfolio.Portfolio_Equity.values
    max_dd_end = np.argmax(np.maximum.accumulate(xs) / xs)
    if max_dd_end == 0:
        max_dd_end = len(xs) - 1
    max_dd_start = np.argmax(xs[:max_dd_end]) if max_dd_end > 0 else 0

    max_ddd_start_day, max_ddd_end_day = max_ddd(xs)
    max_dd_info = "MaxDD  {}~{}, {} days".format(index[max_dd_start].strftime('%Y-%m-%d'),
                                                 index[max_dd_end].strftime('%Y-%m-%d'),
                                                 (index[max_dd_end] - index[max_dd_start]).days)
    max_dd_info += "\nMaxDDD {}~{}, {} days".format(index[max_ddd_start_day].strftime('%Y-%m-%d'),
                                                    index[max_ddd_end_day].strftime('%Y-%m-%d'),
                                                    (index[max_ddd_end_day] - index[max_ddd_start_day]).days)

    plt.style.use('ggplot')

    red = "#aa4643"
    blue = "#4572a7"
    black = "#000000"
    orange = "#FFA500"

    plots_area_size = 0
    if "plots" in result_dict:
        plots_area_size = 8
    img_width = 13
    img_height = 6 + int(plots_area_size * 0.9)

    logo_file = os.path.join(SUPPORT.project_path, 'conf/waterimg.png')
    logo_img = mpimg.imread(logo_file)
    dpi = logo_img.shape[1] / img_width * 1.1

    fig = plt.figure(title, figsize=(20, 15), dpi=160)
    gs = gridspec.GridSpec(10 + plots_area_size, 10)

    # draw risk and portfolio
    font_size = 10
    value_font_size = 10
    label_height, value_height = 0.8, 0.6
    label_height2, value_height2 = 0.35, 0.15

    def _(txt):
        return gettext(txt) if use_chinese_fonts else txt

    fig_data = [
        (0.00, label_height, value_height, _(u"Total Ret"), "{0:.3%}".format(summary["total_returns"]), red, black),
        (0.15, label_height, value_height, _(u"Annual Ret"), "{0:.3%}".format(summary["annualized_returns"]), red, black),
        (0.00, label_height2, value_height2, _(u"Benchmark Ret"), "{0:.3%}".format(summary.get("benchmark_total_returns", 0)), blue,
         black),
        (0.15, label_height2, value_height2, _(u"Benchmark Ann"), "{0:.3%}".format(summary.get("benchmark_annualized_returns", 0)),
         blue, black),

        (0.30, label_height, value_height, _(u"Alpha"), "{0:.4}".format(summary["alpha"]), black, black),
        (0.40, label_height, value_height, _(u"Beta"), "{0:.4}".format(summary["beta"]), black, black),
        (0.55, label_height, value_height, _(u"Sharpe"), "{0:.4}".format(summary["sharpe"]), black, black),
        (0.70, label_height, value_height, _(u"Sortino"), "{0:.4}".format(summary["sortino"]), black, black),
        (0.85, label_height, value_height, _(u"Information Ratio"), "{0:.4}".format(summary["information_ratio"]), black, black),

        (0.30, label_height2, value_height2, _(u"Volatility"), "{0:.4}".format(summary["volatility"]), black, black),
        (0.40, label_height2, value_height2, _(u"MaxDrawdown"), "{0:.3%}".format(summary["max_drawdown"]), black, black),
        (0.55, label_height2, value_height2, _(u"Tracking Error"), "{0:.4}".format(summary["tracking_error"]), black, black),
        (0.70, label_height2, value_height2, _(u"Downside Risk"), "{0:.4}".format(summary["downside_risk"]), black, black),
    ]

    ax = plt.subplot(gs[:3, :-1])
    ax.axis("off")
    for x, y1, y2, label, value, label_color, value_color in fig_data:
        ax.text(x, y1, label, color=label_color, fontsize=font_size)
        ax.text(x, y2, value, color=value_color, fontsize=value_font_size)
    for x, y1, y2, label, value, label_color, value_color in [
        (0.85, label_height2, value_height2, _(u"MaxDD-D"), max_dd_info, black, black)]:
        ax.text(x, y1, label, color=label_color, fontsize=font_size)
        ax.text(x, y2, value, color=value_color, fontsize=8)

    # strategy vs benchmark
    ax = plt.subplot(gs[3:10, :])

    ax.get_xaxis().set_minor_locator(ticker.AutoMinorLocator())
    ax.get_yaxis().set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(b=True, which='minor', linewidth=.2)
    ax.grid(b=True, which='major', linewidth=1)

    # plot two lines
    ax.plot(portfolio.Portfolio_Equity - 1.0, label=_(u"strategy"), alpha=0.6, linewidth=2, color=red)
    if benchmark_portfolio is not None:
        ax.plot(benchmark_portfolio.Benchmark_Equity - 1.0, label=_(u"benchmark"), alpha=0.6, linewidth=2, color=blue)
        ax.plot(portfolio.Portfolio_Equity - benchmark_portfolio.Benchmark_Equity,
                label=_(u"benchmark"), alpha=1, linewidth=1.5, color=orange)

    # plot MaxDD/MaxDDD
    ax.plot([index[max_dd_end], index[max_dd_start]], [rt[max_dd_end] - 1.0, rt[max_dd_start] - 1.0],
            'v', color='Green', markersize=8, alpha=.7, label=_(u"MaxDrawdown"))
    ax.plot([index[max_ddd_start_day], index[max_ddd_end_day]],
            [rt[max_ddd_start_day] - 1.0, rt[max_ddd_end_day] - 1.0], 'D', color='Blue', markersize=8, alpha=.7,
            label=_(u"MaxDDD"))

    # place legend
    leg = plt.legend(loc="best")
    leg.get_frame().set_alpha(0.5)

    # manipulate axis
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:3.2f}%'.format(x * 100) for x in vals])

    # plot user plots
    if "plots" in result_dict:
        plots_df = result_dict["plots"]

        ax2 = plt.subplot(gs[11:, :])
        for column in plots_df.columns:
            ax2.plot(plots_df[column], label=column)

        leg = plt.legend(loc="best")
        leg.get_frame().set_alpha(0.5)

    # logo as watermark
    fig.figimage(
        logo_img,
        xo=((img_width * dpi) - logo_img.shape[1]) / 2,
        yo=(img_height * dpi - logo_img.shape[0]) / 2 + 700,
        alpha=0.6,
    )

    if save_path is not None:
        plt.savefig(save_path)

    if show_windows:
        plt.show()

    return
