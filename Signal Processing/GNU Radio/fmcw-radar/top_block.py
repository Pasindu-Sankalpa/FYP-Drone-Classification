#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Top Block
# GNU Radio version: 3.9.8.0

from distutils.version import StrictVersion

if __name__ == '__main__':
    import ctypes
    import sys
    if sys.platform.startswith('linux'):
        try:
            x11 = ctypes.cdll.LoadLibrary('libX11.so')
            x11.XInitThreads()
        except:
            print("Warning: failed to XInitThreads()")

from PyQt5 import Qt
from gnuradio import qtgui
from gnuradio.filter import firdes
import sip
from gnuradio import analog
from gnuradio import blocks
from gnuradio import fft
from gnuradio.fft import window
from gnuradio import filter
from gnuradio import gr
import sys
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio.qtgui import Range, RangeWidget
from PyQt5 import QtCore



from gnuradio import qtgui

class top_block(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "Top Block", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Top Block")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except:
            pass
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("GNU Radio", "top_block")

        try:
            if StrictVersion(Qt.qVersion()) < StrictVersion("5.0.0"):
                self.restoreGeometry(self.settings.value("geometry").toByteArray())
            else:
                self.restoreGeometry(self.settings.value("geometry"))
        except:
            pass

        ##################################################
        # Variables
        ##################################################
        self.samp_rate = samp_rate = 0.6e6
        self.fft_size = fft_size = 1024*16
        self.delay = delay = 0
        self.ch_freq = ch_freq = 2.2e9
        self.ch_bw = ch_bw = 25e6

        ##################################################
        # Blocks
        ##################################################
        self._delay_range = Range(0, 2000, 100, 0, 2000)
        self._delay_win = RangeWidget(self._delay_range, self.set_delay, "'delay'", "counter_slider", int, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._delay_win)
        self.qtgui_waterfall_sink_x_0_0 = qtgui.waterfall_sink_c(
            2048, #size
            window.WIN_BLACKMAN_hARRIS, #wintype
            0, #fc
            samp_rate, #bw
            'Waterfall Plot', #name
            1, #number of inputs
            None # parent
        )
        self.qtgui_waterfall_sink_x_0_0.set_update_time(0.10)
        self.qtgui_waterfall_sink_x_0_0.enable_grid(False)
        self.qtgui_waterfall_sink_x_0_0.enable_axis_labels(True)



        labels = ['', '', '', '', '',
                  '', '', '', '', '']
        colors = [0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
                  1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_waterfall_sink_x_0_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_waterfall_sink_x_0_0.set_line_label(i, labels[i])
            self.qtgui_waterfall_sink_x_0_0.set_color_map(i, colors[i])
            self.qtgui_waterfall_sink_x_0_0.set_line_alpha(i, alphas[i])

        self.qtgui_waterfall_sink_x_0_0.set_intensity_range(-140, 10)

        self._qtgui_waterfall_sink_x_0_0_win = sip.wrapinstance(self.qtgui_waterfall_sink_x_0_0.qwidget(), Qt.QWidget)

        self.top_layout.addWidget(self._qtgui_waterfall_sink_x_0_0_win)
        self.qtgui_time_sink_x_2 = qtgui.time_sink_c(
            fft_size, #size
            samp_rate, #samp_rate
            'FFT', #name
            1, #number of inputs
            None # parent
        )
        self.qtgui_time_sink_x_2.set_update_time(0.0001)
        self.qtgui_time_sink_x_2.set_y_axis(-3000, 3000)

        self.qtgui_time_sink_x_2.set_y_label('Amplitude', "")

        self.qtgui_time_sink_x_2.enable_tags(True)
        self.qtgui_time_sink_x_2.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, 0, "")
        self.qtgui_time_sink_x_2.enable_autoscale(False)
        self.qtgui_time_sink_x_2.enable_grid(False)
        self.qtgui_time_sink_x_2.enable_axis_labels(True)
        self.qtgui_time_sink_x_2.enable_control_panel(False)
        self.qtgui_time_sink_x_2.enable_stem_plot(False)


        labels = ['Signal 1', 'Signal 2', 'Signal 3', 'Signal 4', 'Signal 5',
            'Signal 6', 'Signal 7', 'Signal 8', 'Signal 9', 'Signal 10']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ['blue', 'red', 'green', 'black', 'cyan',
            'magenta', 'yellow', 'dark red', 'dark green', 'dark blue']
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]
        styles = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        markers = [-1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1]


        for i in range(2):
            if len(labels[i]) == 0:
                if (i % 2 == 0):
                    self.qtgui_time_sink_x_2.set_line_label(i, "Re{{Data {0}}}".format(i/2))
                else:
                    self.qtgui_time_sink_x_2.set_line_label(i, "Im{{Data {0}}}".format(i/2))
            else:
                self.qtgui_time_sink_x_2.set_line_label(i, labels[i])
            self.qtgui_time_sink_x_2.set_line_width(i, widths[i])
            self.qtgui_time_sink_x_2.set_line_color(i, colors[i])
            self.qtgui_time_sink_x_2.set_line_style(i, styles[i])
            self.qtgui_time_sink_x_2.set_line_marker(i, markers[i])
            self.qtgui_time_sink_x_2.set_line_alpha(i, alphas[i])

        self._qtgui_time_sink_x_2_win = sip.wrapinstance(self.qtgui_time_sink_x_2.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_time_sink_x_2_win)
        self.qtgui_time_sink_x_1_1 = qtgui.time_sink_f(
            1024, #size
            samp_rate, #samp_rate
            'RX Power', #name
            1, #number of inputs
            None # parent
        )
        self.qtgui_time_sink_x_1_1.set_update_time(0.10)
        self.qtgui_time_sink_x_1_1.set_y_axis(-0.5, 2.5)

        self.qtgui_time_sink_x_1_1.set_y_label('Watt', "")

        self.qtgui_time_sink_x_1_1.enable_tags(True)
        self.qtgui_time_sink_x_1_1.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, 0, "")
        self.qtgui_time_sink_x_1_1.enable_autoscale(False)
        self.qtgui_time_sink_x_1_1.enable_grid(False)
        self.qtgui_time_sink_x_1_1.enable_axis_labels(True)
        self.qtgui_time_sink_x_1_1.enable_control_panel(False)
        self.qtgui_time_sink_x_1_1.enable_stem_plot(False)


        labels = ['Signal 1', 'Signal 2', 'Signal 3', 'Signal 4', 'Signal 5',
            'Signal 6', 'Signal 7', 'Signal 8', 'Signal 9', 'Signal 10']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ['blue', 'red', 'green', 'black', 'cyan',
            'magenta', 'yellow', 'dark red', 'dark green', 'dark blue']
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]
        styles = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        markers = [-1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1]


        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_time_sink_x_1_1.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_time_sink_x_1_1.set_line_label(i, labels[i])
            self.qtgui_time_sink_x_1_1.set_line_width(i, widths[i])
            self.qtgui_time_sink_x_1_1.set_line_color(i, colors[i])
            self.qtgui_time_sink_x_1_1.set_line_style(i, styles[i])
            self.qtgui_time_sink_x_1_1.set_line_marker(i, markers[i])
            self.qtgui_time_sink_x_1_1.set_line_alpha(i, alphas[i])

        self._qtgui_time_sink_x_1_1_win = sip.wrapinstance(self.qtgui_time_sink_x_1_1.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_time_sink_x_1_1_win)
        self.qtgui_time_sink_x_1 = qtgui.time_sink_f(
            1024, #size
            samp_rate, #samp_rate
            'RX Power', #name
            1, #number of inputs
            None # parent
        )
        self.qtgui_time_sink_x_1.set_update_time(0.10)
        self.qtgui_time_sink_x_1.set_y_axis(0, 20)

        self.qtgui_time_sink_x_1.set_y_label('dBm', "")

        self.qtgui_time_sink_x_1.enable_tags(True)
        self.qtgui_time_sink_x_1.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, 0, "")
        self.qtgui_time_sink_x_1.enable_autoscale(False)
        self.qtgui_time_sink_x_1.enable_grid(False)
        self.qtgui_time_sink_x_1.enable_axis_labels(True)
        self.qtgui_time_sink_x_1.enable_control_panel(False)
        self.qtgui_time_sink_x_1.enable_stem_plot(False)


        labels = ['Signal 1', 'Signal 2', 'Signal 3', 'Signal 4', 'Signal 5',
            'Signal 6', 'Signal 7', 'Signal 8', 'Signal 9', 'Signal 10']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ['blue', 'red', 'green', 'black', 'cyan',
            'magenta', 'yellow', 'dark red', 'dark green', 'dark blue']
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]
        styles = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        markers = [-1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1]


        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_time_sink_x_1.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_time_sink_x_1.set_line_label(i, labels[i])
            self.qtgui_time_sink_x_1.set_line_width(i, widths[i])
            self.qtgui_time_sink_x_1.set_line_color(i, colors[i])
            self.qtgui_time_sink_x_1.set_line_style(i, styles[i])
            self.qtgui_time_sink_x_1.set_line_marker(i, markers[i])
            self.qtgui_time_sink_x_1.set_line_alpha(i, alphas[i])

        self._qtgui_time_sink_x_1_win = sip.wrapinstance(self.qtgui_time_sink_x_1.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_time_sink_x_1_win)
        self.qtgui_time_sink_x_0_0 = qtgui.time_sink_f(
            1024, #size
            samp_rate, #samp_rate
            'Beat Signal', #name
            1, #number of inputs
            None # parent
        )
        self.qtgui_time_sink_x_0_0.set_update_time(0.10)
        self.qtgui_time_sink_x_0_0.set_y_axis(-1.5, 1.5)

        self.qtgui_time_sink_x_0_0.set_y_label('Amplitude', "")

        self.qtgui_time_sink_x_0_0.enable_tags(True)
        self.qtgui_time_sink_x_0_0.set_trigger_mode(qtgui.TRIG_MODE_AUTO, qtgui.TRIG_SLOPE_POS, 0.0, 0, 0, "")
        self.qtgui_time_sink_x_0_0.enable_autoscale(False)
        self.qtgui_time_sink_x_0_0.enable_grid(False)
        self.qtgui_time_sink_x_0_0.enable_axis_labels(True)
        self.qtgui_time_sink_x_0_0.enable_control_panel(False)
        self.qtgui_time_sink_x_0_0.enable_stem_plot(False)


        labels = ['', '', '', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ['blue', 'red', 'green', 'black', 'cyan',
            'magenta', 'yellow', 'dark red', 'dark green', 'dark blue']
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]
        styles = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        markers = [-1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1]


        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_time_sink_x_0_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_time_sink_x_0_0.set_line_label(i, labels[i])
            self.qtgui_time_sink_x_0_0.set_line_width(i, widths[i])
            self.qtgui_time_sink_x_0_0.set_line_color(i, colors[i])
            self.qtgui_time_sink_x_0_0.set_line_style(i, styles[i])
            self.qtgui_time_sink_x_0_0.set_line_marker(i, markers[i])
            self.qtgui_time_sink_x_0_0.set_line_alpha(i, alphas[i])

        self._qtgui_time_sink_x_0_0_win = sip.wrapinstance(self.qtgui_time_sink_x_0_0.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_time_sink_x_0_0_win)
        self.qtgui_time_sink_x_0 = qtgui.time_sink_f(
            1024, #size
            samp_rate, #samp_rate
            'TX/RX Signals', #name
            2, #number of inputs
            None # parent
        )
        self.qtgui_time_sink_x_0.set_update_time(0.10)
        self.qtgui_time_sink_x_0.set_y_axis(-1.5, 1.5)

        self.qtgui_time_sink_x_0.set_y_label('Amplitude', "")

        self.qtgui_time_sink_x_0.enable_tags(True)
        self.qtgui_time_sink_x_0.set_trigger_mode(qtgui.TRIG_MODE_AUTO, qtgui.TRIG_SLOPE_POS, 0.0, 0, 0, "")
        self.qtgui_time_sink_x_0.enable_autoscale(False)
        self.qtgui_time_sink_x_0.enable_grid(False)
        self.qtgui_time_sink_x_0.enable_axis_labels(True)
        self.qtgui_time_sink_x_0.enable_control_panel(False)
        self.qtgui_time_sink_x_0.enable_stem_plot(False)


        labels = ['TX', 'RX', '', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ['blue', 'red', 'green', 'black', 'cyan',
            'magenta', 'yellow', 'dark red', 'dark green', 'dark blue']
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]
        styles = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        markers = [-1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1]


        for i in range(2):
            if len(labels[i]) == 0:
                self.qtgui_time_sink_x_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_time_sink_x_0.set_line_label(i, labels[i])
            self.qtgui_time_sink_x_0.set_line_width(i, widths[i])
            self.qtgui_time_sink_x_0.set_line_color(i, colors[i])
            self.qtgui_time_sink_x_0.set_line_style(i, styles[i])
            self.qtgui_time_sink_x_0.set_line_marker(i, markers[i])
            self.qtgui_time_sink_x_0.set_line_alpha(i, alphas[i])

        self._qtgui_time_sink_x_0_win = sip.wrapinstance(self.qtgui_time_sink_x_0.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_time_sink_x_0_win)
        self.qtgui_time_raster_sink_x_0_0 = qtgui.time_raster_sink_f(
            samp_rate,
            64,
            fft_size,
            [],
            [],
            "Mag",
            1,
            None
        )

        self.qtgui_time_raster_sink_x_0_0.set_update_time(0.10)
        self.qtgui_time_raster_sink_x_0_0.set_intensity_range(-1, 1)
        self.qtgui_time_raster_sink_x_0_0.enable_grid(False)
        self.qtgui_time_raster_sink_x_0_0.enable_axis_labels(True)
        self.qtgui_time_raster_sink_x_0_0.set_x_label("")
        self.qtgui_time_raster_sink_x_0_0.set_x_range(0.0, 0.0)
        self.qtgui_time_raster_sink_x_0_0.set_y_label("")
        self.qtgui_time_raster_sink_x_0_0.set_y_range(0.0, 0.0)

        labels = ['', '', '', '', '',
            '', '', '', '', '']
        colors = [0, 0, 0, 0, 0,
            0, 0, 0, 0, 0]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_time_raster_sink_x_0_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_time_raster_sink_x_0_0.set_line_label(i, labels[i])
            self.qtgui_time_raster_sink_x_0_0.set_color_map(i, colors[i])
            self.qtgui_time_raster_sink_x_0_0.set_line_alpha(i, alphas[i])

        self._qtgui_time_raster_sink_x_0_0_win = sip.wrapinstance(self.qtgui_time_raster_sink_x_0_0.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_time_raster_sink_x_0_0_win)
        self.qtgui_freq_sink_x_1_0 = qtgui.freq_sink_f(
            1024, #size
            window.WIN_HAMMING, #wintype
            0, #fc
            ch_bw, #bw
            'MIX PSD', #name
            1,
            None # parent
        )
        self.qtgui_freq_sink_x_1_0.set_update_time(0.00001)
        self.qtgui_freq_sink_x_1_0.set_y_axis(-80, 20)
        self.qtgui_freq_sink_x_1_0.set_y_label('Relative Gain', 'dBm')
        self.qtgui_freq_sink_x_1_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, 0.0, 0, "")
        self.qtgui_freq_sink_x_1_0.enable_autoscale(False)
        self.qtgui_freq_sink_x_1_0.enable_grid(False)
        self.qtgui_freq_sink_x_1_0.set_fft_average(1.0)
        self.qtgui_freq_sink_x_1_0.enable_axis_labels(True)
        self.qtgui_freq_sink_x_1_0.enable_control_panel(False)
        self.qtgui_freq_sink_x_1_0.set_fft_window_normalized(False)


        self.qtgui_freq_sink_x_1_0.set_plot_pos_half(not True)

        labels = ['', '', '', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["blue", "red", "green", "black", "cyan",
            "magenta", "yellow", "dark red", "dark green", "dark blue"]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_freq_sink_x_1_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_freq_sink_x_1_0.set_line_label(i, labels[i])
            self.qtgui_freq_sink_x_1_0.set_line_width(i, widths[i])
            self.qtgui_freq_sink_x_1_0.set_line_color(i, colors[i])
            self.qtgui_freq_sink_x_1_0.set_line_alpha(i, alphas[i])

        self._qtgui_freq_sink_x_1_0_win = sip.wrapinstance(self.qtgui_freq_sink_x_1_0.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_freq_sink_x_1_0_win)
        self.qtgui_freq_sink_x_1 = qtgui.freq_sink_c(
            1024, #size
            window.WIN_BLACKMAN_hARRIS, #wintype
            ch_freq, #fc
            ch_bw, #bw
            'RX PSD', #name
            1,
            None # parent
        )
        self.qtgui_freq_sink_x_1.set_update_time(0.00001)
        self.qtgui_freq_sink_x_1.set_y_axis(-80, 20)
        self.qtgui_freq_sink_x_1.set_y_label('Relative Gain', 'dBm')
        self.qtgui_freq_sink_x_1.set_trigger_mode(qtgui.TRIG_MODE_FREE, 0.0, 0, "")
        self.qtgui_freq_sink_x_1.enable_autoscale(False)
        self.qtgui_freq_sink_x_1.enable_grid(False)
        self.qtgui_freq_sink_x_1.set_fft_average(1.0)
        self.qtgui_freq_sink_x_1.enable_axis_labels(True)
        self.qtgui_freq_sink_x_1.enable_control_panel(False)
        self.qtgui_freq_sink_x_1.set_fft_window_normalized(False)



        labels = ['', '', '', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["blue", "red", "green", "black", "cyan",
            "magenta", "yellow", "dark red", "dark green", "dark blue"]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_freq_sink_x_1.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_freq_sink_x_1.set_line_label(i, labels[i])
            self.qtgui_freq_sink_x_1.set_line_width(i, widths[i])
            self.qtgui_freq_sink_x_1.set_line_color(i, colors[i])
            self.qtgui_freq_sink_x_1.set_line_alpha(i, alphas[i])

        self._qtgui_freq_sink_x_1_win = sip.wrapinstance(self.qtgui_freq_sink_x_1.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_freq_sink_x_1_win)
        self.qtgui_freq_sink_x_0 = qtgui.freq_sink_c(
            1024, #size
            window.WIN_BLACKMAN_hARRIS, #wintype
            0, #fc
            ch_bw, #bw
            'TX PSD', #name
            1,
            None # parent
        )
        self.qtgui_freq_sink_x_0.set_update_time(0.00001)
        self.qtgui_freq_sink_x_0.set_y_axis(-140, 10)
        self.qtgui_freq_sink_x_0.set_y_label('Relative Gain', 'dBm')
        self.qtgui_freq_sink_x_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, 0.0, 0, "")
        self.qtgui_freq_sink_x_0.enable_autoscale(False)
        self.qtgui_freq_sink_x_0.enable_grid(False)
        self.qtgui_freq_sink_x_0.set_fft_average(1.0)
        self.qtgui_freq_sink_x_0.enable_axis_labels(True)
        self.qtgui_freq_sink_x_0.enable_control_panel(False)
        self.qtgui_freq_sink_x_0.set_fft_window_normalized(False)



        labels = ['', '', '', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["blue", "red", "green", "black", "cyan",
            "magenta", "yellow", "dark red", "dark green", "dark blue"]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_freq_sink_x_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_freq_sink_x_0.set_line_label(i, labels[i])
            self.qtgui_freq_sink_x_0.set_line_width(i, widths[i])
            self.qtgui_freq_sink_x_0.set_line_color(i, colors[i])
            self.qtgui_freq_sink_x_0.set_line_alpha(i, alphas[i])

        self._qtgui_freq_sink_x_0_win = sip.wrapinstance(self.qtgui_freq_sink_x_0.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_freq_sink_x_0_win)
        self.low_pass_filter_0 = filter.fir_filter_fff(
            1,
            firdes.low_pass(
                1,
                samp_rate,
                100000,
                10000,
                window.WIN_RECTANGULAR,
                6.76))
        self.high_pass_filter_0 = filter.fir_filter_ccf(
            1,
            firdes.high_pass(
                1,
                samp_rate,
                1e3,
                1e3,
                window.WIN_HAMMING,
                6.76))
        self.fft_vxx_0_0_0_1 = fft.fft_vcc(fft_size, True, window.blackmanharris(fft_size), True, 8)
        self.blocks_vector_to_stream_0_0_0 = blocks.vector_to_stream(gr.sizeof_gr_complex*1, fft_size)
        self.blocks_vector_to_stream_0_0 = blocks.vector_to_stream(gr.sizeof_float*1, fft_size)
        self.blocks_vco_c_0_0 = blocks.vco_c(samp_rate, samp_rate, 1)
        self.blocks_stream_to_vector_0_1 = blocks.stream_to_vector(gr.sizeof_gr_complex*1, fft_size)
        self.blocks_nlog10_ff_0 = blocks.nlog10_ff(10, 1, 0)
        self.blocks_multiply_conjugate_cc_0 = blocks.multiply_conjugate_cc(1)
        self.blocks_moving_average_xx_0 = blocks.moving_average_ff(50, 1, 4000, 1)
        self.blocks_delay_0 = blocks.delay(gr.sizeof_gr_complex*1, delay)
        self.blocks_complex_to_real_0_0_0 = blocks.complex_to_real(1)
        self.blocks_complex_to_real_0_0 = blocks.complex_to_real(1)
        self.blocks_complex_to_real_0 = blocks.complex_to_real(1)
        self.blocks_complex_to_mag_squared_0 = blocks.complex_to_mag_squared(1)
        self.blocks_complex_to_mag_0_0 = blocks.complex_to_mag(fft_size)
        self.analog_sig_source_x_0_0 = analog.sig_source_f(samp_rate, analog.GR_SAW_WAVE, 1000, 1, 0, 0)
        self.analog_agc_xx_0_0 = analog.agc_cc(1e-4, 1.0, 1.0)
        self.analog_agc_xx_0_0.set_max_gain(65536)
        self.analog_agc_xx_0 = analog.agc_cc(1e-4, 1.0, 1.0)
        self.analog_agc_xx_0.set_max_gain(65536)


        ##################################################
        # Connections
        ##################################################
        self.connect((self.analog_agc_xx_0, 0), (self.blocks_complex_to_real_0, 0))
        self.connect((self.analog_agc_xx_0, 0), (self.blocks_multiply_conjugate_cc_0, 0))
        self.connect((self.analog_agc_xx_0_0, 0), (self.blocks_complex_to_real_0_0, 0))
        self.connect((self.analog_agc_xx_0_0, 0), (self.blocks_multiply_conjugate_cc_0, 1))
        self.connect((self.analog_agc_xx_0_0, 0), (self.qtgui_freq_sink_x_1, 0))
        self.connect((self.analog_sig_source_x_0_0, 0), (self.blocks_vco_c_0_0, 0))
        self.connect((self.blocks_complex_to_mag_0_0, 0), (self.blocks_vector_to_stream_0_0, 0))
        self.connect((self.blocks_complex_to_mag_squared_0, 0), (self.blocks_moving_average_xx_0, 0))
        self.connect((self.blocks_complex_to_mag_squared_0, 0), (self.qtgui_time_sink_x_1_1, 0))
        self.connect((self.blocks_complex_to_real_0, 0), (self.qtgui_time_sink_x_0, 0))
        self.connect((self.blocks_complex_to_real_0_0, 0), (self.qtgui_time_sink_x_0, 1))
        self.connect((self.blocks_complex_to_real_0_0_0, 0), (self.low_pass_filter_0, 0))
        self.connect((self.blocks_complex_to_real_0_0_0, 0), (self.qtgui_time_sink_x_0_0, 0))
        self.connect((self.blocks_delay_0, 0), (self.analog_agc_xx_0_0, 0))
        self.connect((self.blocks_moving_average_xx_0, 0), (self.blocks_nlog10_ff_0, 0))
        self.connect((self.blocks_multiply_conjugate_cc_0, 0), (self.blocks_complex_to_mag_squared_0, 0))
        self.connect((self.blocks_multiply_conjugate_cc_0, 0), (self.blocks_complex_to_real_0_0_0, 0))
        self.connect((self.blocks_multiply_conjugate_cc_0, 0), (self.blocks_stream_to_vector_0_1, 0))
        self.connect((self.blocks_multiply_conjugate_cc_0, 0), (self.high_pass_filter_0, 0))
        self.connect((self.blocks_nlog10_ff_0, 0), (self.qtgui_time_sink_x_1, 0))
        self.connect((self.blocks_stream_to_vector_0_1, 0), (self.fft_vxx_0_0_0_1, 0))
        self.connect((self.blocks_vco_c_0_0, 0), (self.analog_agc_xx_0, 0))
        self.connect((self.blocks_vco_c_0_0, 0), (self.blocks_delay_0, 0))
        self.connect((self.blocks_vco_c_0_0, 0), (self.qtgui_freq_sink_x_0, 0))
        self.connect((self.blocks_vector_to_stream_0_0, 0), (self.qtgui_time_raster_sink_x_0_0, 0))
        self.connect((self.blocks_vector_to_stream_0_0_0, 0), (self.qtgui_time_sink_x_2, 0))
        self.connect((self.fft_vxx_0_0_0_1, 0), (self.blocks_complex_to_mag_0_0, 0))
        self.connect((self.fft_vxx_0_0_0_1, 0), (self.blocks_vector_to_stream_0_0_0, 0))
        self.connect((self.high_pass_filter_0, 0), (self.qtgui_waterfall_sink_x_0_0, 0))
        self.connect((self.low_pass_filter_0, 0), (self.qtgui_freq_sink_x_1_0, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "top_block")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.analog_sig_source_x_0_0.set_sampling_freq(self.samp_rate)
        self.high_pass_filter_0.set_taps(firdes.high_pass(1, self.samp_rate, 1e3, 1e3, window.WIN_HAMMING, 6.76))
        self.low_pass_filter_0.set_taps(firdes.low_pass(1, self.samp_rate, 100000, 10000, window.WIN_RECTANGULAR, 6.76))
        self.qtgui_time_sink_x_0.set_samp_rate(self.samp_rate)
        self.qtgui_time_sink_x_0_0.set_samp_rate(self.samp_rate)
        self.qtgui_time_sink_x_1.set_samp_rate(self.samp_rate)
        self.qtgui_time_sink_x_1_1.set_samp_rate(self.samp_rate)
        self.qtgui_time_sink_x_2.set_samp_rate(self.samp_rate)
        self.qtgui_waterfall_sink_x_0_0.set_frequency_range(0, self.samp_rate)

    def get_fft_size(self):
        return self.fft_size

    def set_fft_size(self, fft_size):
        self.fft_size = fft_size
        self.qtgui_time_raster_sink_x_0_0.set_num_cols(self.fft_size)

    def get_delay(self):
        return self.delay

    def set_delay(self, delay):
        self.delay = delay
        self.blocks_delay_0.set_dly(self.delay)

    def get_ch_freq(self):
        return self.ch_freq

    def set_ch_freq(self, ch_freq):
        self.ch_freq = ch_freq
        self.qtgui_freq_sink_x_1.set_frequency_range(self.ch_freq, self.ch_bw)

    def get_ch_bw(self):
        return self.ch_bw

    def set_ch_bw(self, ch_bw):
        self.ch_bw = ch_bw
        self.qtgui_freq_sink_x_0.set_frequency_range(0, self.ch_bw)
        self.qtgui_freq_sink_x_1.set_frequency_range(self.ch_freq, self.ch_bw)
        self.qtgui_freq_sink_x_1_0.set_frequency_range(0, self.ch_bw)




def main(top_block_cls=top_block, options=None):

    if StrictVersion("4.5.0") <= StrictVersion(Qt.qVersion()) < StrictVersion("5.0.0"):
        style = gr.prefs().get_string('qtgui', 'style', 'raster')
        Qt.QApplication.setGraphicsSystem(style)
    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls()

    tb.start()

    tb.show()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    qapp.exec_()

if __name__ == '__main__':
    main()
