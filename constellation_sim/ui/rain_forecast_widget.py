"""Weather-style preview of the simulated rain field and online forecasts."""
from __future__ import annotations

import csv
import calendar
import json
import math
import threading
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode

from PySide6.QtCore import QDateTime, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (QComboBox, QDateTimeEdit, QFileDialog, QGridLayout, QHBoxLayout, QLabel,
    QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
    QHeaderView)

from constellation_sim.algorithm.rain_model import (synthetic_surface_rain_rate_mmh,
    synthetic_rain_attenuation_db,
    forecast_rain_rate_mmh, forecast_rain_attenuation_db)


class RainRateChart(QWidget):
    """Station rain-rate curves; samples are shared with the table below."""
    point_clicked=Signal(object)
    COLORS=("#43d5f5", "#f7bd60", "#aa9afa", "#69dbac", "#ff8095", "#f5e37a")

    def __init__(self,parent=None):
        super().__init__(parent)
        self.names=[];self.minutes=[];self.rates=[];self.unit="mm/h";self.monthly=False;self.series_start=None
        self.setMouseTracking(True)
        self.setMinimumHeight(190)

    def set_series(self,names,minutes,rates,unit="mm/h",monthly=False):
        self.names=list(names);self.minutes=list(minutes);self.rates=[list(row) for row in rates]
        self.unit=unit;self.monthly=monthly
        if not monthly:self.series_start=None
        self.update()

    def paintEvent(self,event):
        painter=QPainter(self);painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(),QColor("#081b29"))
        if not self.minutes or not self.rates:painter.end();return
        left,top,right,bottom=48,30,self.width()-18,self.height()-32
        if right<=left or bottom<=top:painter.end();return
        peak=max(1.,max((max(row) for row in self.rates if row),default=0.)*1.1)
        painter.setPen(QPen(QColor("#426577"),1))
        painter.drawLine(left,bottom,right,bottom);painter.drawLine(left,top,left,bottom)
        painter.setPen(QColor("#b8d4e0"))
        painter.drawText(5,top+5,f"{peak:.1f}");painter.drawText(7,bottom,f"0")
        if self.monthly:
            days=max(1,len(self.minutes)//48)
            for day in range(1,days+1,5):
                x=left+(right-left)*(day-1)/days
                painter.drawText(int(x)-10,bottom+18,f"{day:02d}日")
        else:
            painter.drawText(left,bottom+18,f"{self.minutes[0]:g} min")
            painter.drawText(right-52,bottom+18,f"{self.minutes[-1]:g} min")
        painter.drawText(5,17,self.unit)
        span=max(1.,self.minutes[-1]-self.minutes[0])
        for index,row in enumerate(self.rates):
            color=QColor(self.COLORS[index%len(self.COLORS)])
            painter.setPen(QPen(color,2))
            points=[(left+(right-left)*(minute-self.minutes[0])/span,
                     bottom-(bottom-top)*value/peak) for minute,value in zip(self.minutes,row)]
            for first,second in zip(points,points[1:]):
                painter.drawLine(int(first[0]),int(first[1]),int(second[0]),int(second[1]))
            painter.setPen(color)
            painter.drawText(left+index*120,22,self.names[index][:8])
        painter.end()

    def mouseMoveEvent(self,event):
        if not self.monthly or not self.series_start or not self.minutes:return
        left,right=48,self.width()-18
        if right<=left:return
        index=round((event.position().x()-left)/(right-left)*(len(self.minutes)-1))
        index=min(max(index,0),len(self.minutes)-1)
        instant=self.series_start+timedelta(minutes=30*index)
        values="  ".join(f"{name} {row[index]:.2f}" for name,row in zip(self.names,self.rates) if index<len(row))
        self.setToolTip(f"{instant:%Y-%m-%d %H:%M} UTC\n{values} mm/h")

    def mousePressEvent(self,event):
        if not self.monthly or not self.series_start or not self.minutes or event.button()!=Qt.MouseButton.LeftButton:return
        left,right=48,self.width()-18
        if right<=left or not left<=event.position().x()<=right:return
        index=round((event.position().x()-left)/(right-left)*(len(self.minutes)-1))
        index=min(max(index,0),len(self.minutes)-1)
        self.point_clicked.emit(self.series_start+timedelta(minutes=30*index))


def _rain_description(rate):
    if rate < .1:return "☀ 无雨", "#32718c"
    if rate < 2.5:return "☁ 小雨", "#187b9a"
    if rate < 10:return "☂ 中雨", "#99721d"
    return "☂ 强雨", "#9b4350"


class RainForecastWidget(QWidget):
    forecast_loaded=Signal(object)
    online_result=Signal(object,object)
    month_time_selected=Signal(object)
    def __init__(self,parent=None):
        super().__init__(parent)
        self.stations=[];self.duration_s=1800;self.epoch_iso="2000-01-01T00:00:00+00:00";self._reply=None;self.forecast_snapshot=None;self.history_mode=True;self._chart_minutes=[]
        self.station_grades={"拉萨":"中雨","三亚":"重雨"}
        self.forecast_grades={};self.inputs_editable=True
        self.online_result.connect(self._forecast_received)
        layout=QVBoxLayout(self);layout.setContentsMargins(8,8,8,8);layout.setSpacing(6)
        self.description=QLabel("按站点和日期合成降雨；偏低／基准／偏高表示雨量情景，链路雨衰按沿途雨场计算。")
        self.description.setWordWrap(True)
        self.description.hide()
        self.time_context=QLabel();self.time_context.setWordWrap(True)
        self.time_context.setStyleSheet("background:#103348;color:#d9f6ff;border-left:4px solid #2bd3ee;padding:9px 12px;font-weight:600")
        self.time_context.hide()
        self.input_header=QWidget(self);self.input_header.setFixedHeight(36)
        self.input_row=QHBoxLayout(self.input_header);self.input_row.setContentsMargins(0,0,0,0);self.input_row.setSpacing(8)
        layout.addWidget(self.input_header)
        self.time_controls=QWidget(self);controls=QGridLayout(self.time_controls)
        self.time_controls.setFixedHeight(30)
        controls.setContentsMargins(0,0,0,0);controls.setHorizontalSpacing(8);controls.setVerticalSpacing(4)
        layout.addWidget(self.time_controls)
        self.source=QComboBox(self);self.source.addItems(["站点雨衰场景","天气预报"]);self.source.hide()
        self.source.currentIndexChanged.connect(self._source_changed)
        controls.addWidget(QLabel("起点"),0,0)
        self.start_min=QSpinBox();self.start_min.setSuffix(" min");self.start_min.valueChanged.connect(self.refresh)
        self.start_min.setFixedWidth(98);controls.addWidget(self.start_min,0,1)
        controls.addWidget(QLabel("终点"),0,2)
        self.end_min=QSpinBox();self.end_min.setSuffix(" min");self.end_min.valueChanged.connect(self.refresh)
        self.end_min.setFixedWidth(98);controls.addWidget(self.end_min,0,3)
        controls.addWidget(QLabel("采样间隔"),0,4)
        self.interval=QComboBox();self.interval.addItems(["每 1 分钟","每 5 分钟","每 10 分钟"])
        self.interval.setFixedWidth(106)
        self.interval.currentIndexChanged.connect(self.refresh);controls.addWidget(self.interval,0,5)
        controls.addWidget(QLabel("参考仰角"),0,6)
        self.reference_elevation=QSpinBox();self.reference_elevation.setRange(5,90)
        self.reference_elevation.setValue(45);self.reference_elevation.setSuffix("°")
        self.reference_elevation.valueChanged.connect(lambda:self._render_snapshot() if self.source.currentIndex()==1 else self.refresh())
        self.reference_elevation.setFixedWidth(72);controls.addWidget(self.reference_elevation,0,7)
        controls.setColumnStretch(8,1)
        self.station_grade_panel=QWidget(self)
        self.station_grade_panel.setFixedHeight(46)
        self.station_grade_row=QGridLayout(self.station_grade_panel)
        self.station_grade_row.setContentsMargins(0,0,0,0)
        self.station_grade_row.setHorizontalSpacing(10)
        self.station_grade_row.setVerticalSpacing(4)
        layout.addWidget(self.station_grade_panel)
        self.import_button=QPushButton("导入天气预报 CSV")
        self.import_button.setToolTip("CSV 列：station,time_utc,rain_mm_h；time_utc 是雨强时段的结束时刻。")
        self.import_button.clicked.connect(self.import_forecast_csv)
        self.online_controls=QWidget(self);self.online_controls.setFixedHeight(34)
        online=QHBoxLayout(self.online_controls);online.setContentsMargins(0,0,0,0)
        online.addWidget(QLabel("预报起点"))
        self.online_start=QDateTimeEdit(QDateTime.currentDateTime());self.online_start.setDisplayFormat("yyyy-MM-dd HH:mm")
        self.online_start.setReadOnly(True);self.online_start.setButtonSymbols(QDateTimeEdit.ButtonSymbols.NoButtons)
        self.online_start.dateTimeChanged.connect(self._update_time_context)
        online.addWidget(self.online_start)
        self.fetch_button=QPushButton("获取预报");self.fetch_button.clicked.connect(self.fetch_online)
        online.addWidget(self.fetch_button);online.addWidget(self.import_button);online.addStretch(1)
        self.online_controls.hide()
        self.notice=QLabel();self.notice.setWordWrap(True);self.notice.hide();layout.addWidget(self.notice)
        self.data_status=QLabel();layout.addWidget(self.data_status)
        self.cards=QWidget();self.cards.setFixedHeight(78);self.cards_layout=QHBoxLayout(self.cards);self.cards_layout.setContentsMargins(0,0,0,0);layout.addWidget(self.cards)
        self.chart=RainRateChart(self);layout.addWidget(self.chart)
        self.table=QTableWidget(0,0);self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setMaximumHeight(240)
        self.table.verticalHeader().setVisible(False);self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        layout.addWidget(self.table,1)
        self.empty_state=QLabel(self)
        self.empty_state.setFixedHeight(46)
        self.empty_state.setStyleSheet("color:#9ec0cf;background:#103044;border-left:3px solid #20bedb;padding:10px 12px")
        layout.addWidget(self.empty_state)
        self.empty_fill=QWidget(self);layout.addWidget(self.empty_fill,1)
        self.month_toggle=QPushButton("▼ 收起月度雨强（每 30 分钟）")
        self.month_toggle.setCheckable(True)
        self.month_toggle.setChecked(True)
        self.month_toggle.setStyleSheet("QPushButton{background:#10384b;color:#d9f6ff;border-left:4px solid #27d6f3;text-align:left;padding:8px;font-weight:700}")
        self.month_toggle.clicked.connect(self._toggle_monthly)
        layout.addWidget(self.month_toggle)
        self.month_panel=QWidget(self)
        month_layout=QVBoxLayout(self.month_panel)
        month_layout.setContentsMargins(0,4,0,0);month_layout.setSpacing(6)
        month_controls=QHBoxLayout()
        month_controls.addWidget(QLabel("月度背景 · 每 30 分钟雨强"))
        self.month_selector=QComboBox();self.month_selector.addItems([f"{month:02d}月" for month in range(1,13)])
        self.month_selector.setFixedWidth(92)
        self.month_selector.currentIndexChanged.connect(self._render_monthly)
        month_controls.addWidget(self.month_selector)
        month_controls.addStretch(1)
        month_layout.addLayout(month_controls)
        self.month_status=QLabel();month_layout.addWidget(self.month_status)
        self.month_chart=RainRateChart(self.month_panel)
        self.month_chart.setMinimumHeight(340)
        self.month_chart.setCursor(Qt.CursorShape.PointingHandCursor)
        self.month_chart.point_clicked.connect(self.month_time_selected.emit)
        month_layout.addWidget(self.month_chart)
        layout.addWidget(self.month_panel)
        self.month_panel.hide()
        self.end_min.setValue(30)
        self._source_changed()

    def set_scenario(self,stations,duration_s,epoch_iso=None):
        self.stations=[dict(station) for station in (stations or [])];self.duration_s=max(0.,float(duration_s))
        while self.station_grade_row.count():
            item=self.station_grade_row.takeAt(0)
            if item.widget():item.widget().deleteLater()
        for column,station in enumerate(self.stations):
            name=str(station.get("name",""))
            station_label=QLabel(name)
            station_label.setStyleSheet("color:#a9d9e9")
            self.station_grade_row.addWidget(station_label,0,column)
            selector=QComboBox(self)
            selector.setFixedWidth(112)
            selector.setToolTip("雨量情景：偏低／基准／偏高分别取原雨强的 80%／100%／120%；原值为零时也有不同幅度的情景雨强。")
            for label,key in (("偏低", "轻雨"),("基准", "中雨"),("偏高", "重雨")):
                selector.addItem(label,key)
            grades=self.selected_grades()
            saved=grades.get(name,(station.get("rain_grade") or "轻雨") if self.source.currentIndex()==0 else "中雨")
            index=selector.findData(saved)
            selector.setCurrentIndex(max(0,index))
            grades[name]=selector.currentData()
            selector.setEnabled(self.inputs_editable and not self.history_mode)
            selector.currentIndexChanged.connect(lambda _index,n=name,s=selector:self._station_grade_changed(n,s.currentData()))
            self.station_grade_row.addWidget(selector,1,column)
        self.station_grade_row.setColumnStretch(len(self.stations),1)
        if epoch_iso is not None:
            self.epoch_iso=str(epoch_iso)
            if self.forecast_snapshot and datetime.fromisoformat(self.forecast_snapshot.get("start_iso","2000-01-01T00:00:00+00:00"))!=datetime.fromisoformat(self.epoch_iso):
                self.forecast_snapshot=None
        if getattr(self,"_month_epoch",None)!=self.epoch_iso:
            self._month_epoch=self.epoch_iso
            self.month_selector.blockSignals(True)
            self.month_selector.setCurrentIndex(datetime.fromisoformat(self.epoch_iso).month-1)
            self.month_selector.blockSignals(False)
        for station in self.stations:station["simulation_start_utc"]=self.epoch_iso
        last=max(0,int(self.duration_s//60));self.start_min.setMaximum(last);self.end_min.setMaximum(last)
        if self.end_min.value()==0 or self.end_min.value()>last:self.end_min.setValue(last)
        self._update_time_context()
        if self.source.currentIndex()==1:self._render_snapshot()
        else:self.refresh()

    def attach_controls(self,enabled,source):
        enabled.setParent(self)
        enabled.setChecked(True)
        enabled.hide()
        self.input_row.addWidget(QLabel("数据来源"))
        source.setFixedWidth(205)
        self.input_row.addWidget(source)
        self.input_row.addWidget(self.online_controls)
        self.input_row.addStretch(1)

    def set_input_enabled(self,enabled):
        self.inputs_editable=bool(enabled)
        self.fetch_button.setEnabled(enabled)
        self.import_button.setEnabled(enabled)
        for index in range(self.station_grade_row.count()):
            widget=self.station_grade_row.itemAt(index).widget()
            if isinstance(widget,QComboBox):widget.setEnabled(enabled and not self.history_mode)

    def selected_grades(self):
        return self.forecast_grades if self.source.currentIndex()==1 else self.station_grades

    def _station_grade_changed(self,name,grade):
        self.selected_grades()[name]=grade
        if self.source.currentIndex()==1:self._render_snapshot()
        else:self.refresh()

    def _update_time_context(self,*_args):
        try:epoch=datetime.fromisoformat(self.epoch_iso)
        except ValueError:epoch=None
        if epoch:
            sim_end=epoch+timedelta(seconds=self.duration_s)
            orbit=f"轨道仿真：{epoch:%Y-%m-%d %H:%M:%S} 至 {sim_end:%H:%M:%S} UTC（总长 {self.duration_s/60:g} 分钟）"
        else:orbit=f"轨道仿真历元：生成基础数据后确定（总长 {self.duration_s/60:g} 分钟）"
        if self.source.currentIndex()==1:
            weather=self.online_start.dateTime().toPython()
            mapped=f"天气预报：{weather:%Y-%m-%d %H:%M} 起，与轨道仿真 t=0 使用同一绝对时间。"
        else:
            if epoch:
                start=epoch+timedelta(minutes=self.start_min.value())
                end=epoch+timedelta(minutes=max(self.start_min.value(),self.end_min.value()))
                mapped=f"当前雨衰：{start:%Y-%m-%d %H:%M} 至 {end:%H:%M} UTC（t={self.start_min.value()}–{max(self.start_min.value(),self.end_min.value())} 分钟）"
            else:mapped=f"当前雨衰：轨道仿真 t={self.start_min.value()}–{max(self.start_min.value(),self.end_min.value())} 分钟"
        self.time_context.setText(orbit+"\n"+mapped)

    def _render_snapshot(self):
        snapshot=self.forecast_snapshot
        if not snapshot:return
        records=snapshot.get("stations") or []
        if not records:return
        start=snapshot.get("start_iso","")
        if datetime.fromisoformat(start)!=datetime.fromisoformat(self.epoch_iso):
            self._clear("仿真起点已改变，请重新获取天气预报")
            return
        stations=[{"name":row.get("name",f"站{index+1}")} for index,row in enumerate(records)]
        epoch=datetime.fromisoformat(start)
        last=max(0,int(self.duration_s//60))
        minutes=list(range(last+1))
        if len(minutes)>61:minutes=minutes[::max(1,math.ceil(len(minutes)/61))][:61]
        if minutes[-1]!=last:minutes.append(last)
        times=[f"{minute//60:02d}:{minute%60:02d}" for minute in minutes]
        rates=[]
        for row in records:
            site=next((dict(item) for item in self.stations if item.get("name")==row.get("name")),{"name":row.get("name","")})
            site["simulation_start_utc"]=self.epoch_iso
            ends=[(datetime.fromisoformat(value)-epoch).total_seconds() for value in row["times"]]
            if not ends or not any(0<=value<=3600 for value in ends) or ends[-1]<self.duration_s:
                self.forecast_snapshot=None
                self._clear(f"{site['name']} 的预报没有覆盖本次仿真时段，请重新获取")
                return
            site["forecast_rain"]={"hour_ends_s":ends,"rates_mmh":row["rates"]}
            site["forecast_rain_scale"]={"轻雨":.8,"中雨":1.,"重雨":1.2}.get(self.forecast_grades.get(row.get("name"),"中雨"),1.)
            try:rates.append([forecast_rain_rate_mmh(site,minute*60.) for minute in minutes])
            except ValueError as exc:
                self._clear(str(exc));return
        self._chart_minutes=minutes
        self._show(stations,times,rates,
            f"天气预报 · 本次仿真 {last} 分钟 · 起点 {epoch.astimezone(timezone(timedelta(hours=8))):%Y-%m-%d %H:%M}（北京时间）")

    def _source_changed(self):
        online=self.source.currentIndex()==1
        monthly_available=not online and not self.history_mode
        self.month_toggle.setVisible(monthly_available)
        self.month_panel.setVisible(monthly_available and self.month_toggle.isChecked())
        self.online_controls.setVisible(online)
        self.time_controls.setVisible(not online)
        self.data_status.setVisible(not online)
        for index in range(self.station_grade_row.count()):
            widget=self.station_grade_row.itemAt(index).widget()
            if isinstance(widget,QComboBox):widget.setEnabled(self.inputs_editable and not self.history_mode)
        for widget in (self.start_min,self.end_min,self.interval):widget.setEnabled(not online)
        self.description.setText("偏低／基准／偏高表示雨量情景；原雨强为零时仍有不同幅度的情景雨强。情景值未经实测标定。")
        if online:
            if self.forecast_snapshot:self._render_snapshot()
            else:self._clear("")
        else:self.refresh()
        self._update_time_context()

    def _clear(self,message):
        self.notice.setText(message);self.notice.setVisible(bool(message));self.table.setRowCount(0);self.table.setColumnCount(0);self.chart.set_series([],[],[])
        self.cards.hide();self.chart.hide();self.table.hide()
        self.empty_state.setText(message or "获取或导入天气预报后，将显示本次仿真时段的站点雨强。")
        self.empty_state.show();self.empty_fill.show()
        if message:self.data_status.setText("")
        while self.cards_layout.count():
            item=self.cards_layout.takeAt(0)
            if item.widget():item.widget().deleteLater()

    def _show(self,stations,times,rates,source_label):
        self._clear("")
        self.cards.show()
        self.chart.show()
        self.table.show()
        self.empty_state.hide();self.empty_fill.hide()
        self.data_status.setText(source_label)
        if not stations:
            self.notice.setText("暂无信关站");self.notice.show()
            return
        for index,station in enumerate(stations[:6]):
            rate=float(rates[index][0]) if rates[index] else 0.
            condition,color=_rain_description(rate)
            site=next((dict(row) for row in self.stations if row.get("name")==station.get("name")),None)
            fade=None
            if site is not None:
                site["simulation_start_utc"]=self.epoch_iso
                try:
                    if self.source.currentIndex()==1 and self.forecast_snapshot:
                        record=next(row for row in self.forecast_snapshot["stations"] if row["name"]==site["name"])
                        epoch=datetime.fromisoformat(self.forecast_snapshot["start_iso"])
                        site["forecast_rain"]={"hour_ends_s":[(datetime.fromisoformat(value)-epoch).total_seconds() for value in record["times"]],"rates_mmh":record["rates"]}
                        site["forecast_rain_scale"]={"轻雨":.8,"中雨":1.,"重雨":1.2}.get(self.forecast_grades.get(site["name"],"中雨"),1.)
                        fade=forecast_rain_attenuation_db(site,self.reference_elevation.value(),0.)
                    else:
                        site["rain_grade"]=self.station_grades.get(site["name"],"轻雨")
                        fade=synthetic_rain_attenuation_db(site,self.reference_elevation.value(),"轻雨",self.start_min.value()*60.)
                except (KeyError,ValueError,StopIteration):pass
            fade_text=f" · {fade:.2f} dB" if fade is not None else ""
            card=QLabel(f"{station.get('name','信关站')}\n{rate:.2f} mm/h{fade_text}")
            card.setAlignment(Qt.AlignmentFlag.AlignCenter)
            card.setMinimumHeight(78)
            card.setStyleSheet(f"background:{color};border:1px solid #48a9bf;border-radius:8px;color:white;font-size:13px;font-weight:600")
            self.cards_layout.addWidget(card)
        self.cards_layout.addStretch(1)
        self.table.setRowCount(len(stations));self.table.setColumnCount(len(times)+1)
        self.table.setHorizontalHeaderLabels(["信关站"]+times)
        self.table.setColumnWidth(0,150)
        for column in range(1,len(times)+1):self.table.setColumnWidth(column,112)
        for row,station in enumerate(stations):
            self.table.setItem(row,0,QTableWidgetItem(str(station.get("name",f"站{row+1}"))))
            for column,rate in enumerate(rates[row],1):
                condition,color=_rain_description(float(rate))
                cell=QTableWidgetItem(f"{float(rate):.2f} mm/h")
                cell.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                cell.setBackground(QColor(color));self.table.setItem(row,column,cell)
        self.table.resizeRowsToContents()
        self.chart.set_series([str(station.get("name",f"站点{index+1}")) for index,station in enumerate(stations)],self._chart_minutes,rates)

    def _render_monthly(self):
        if not self.stations or not self.month_panel.isVisible():return
        epoch=datetime.fromisoformat(self.epoch_iso)
        if epoch.tzinfo is None:epoch=epoch.replace(tzinfo=timezone.utc)
        year=epoch.year;month=self.month_selector.currentIndex()+1
        month_start=datetime(year,month,1,tzinfo=timezone.utc)
        first_s=(month_start-epoch.astimezone(timezone.utc)).total_seconds()
        sample_count=48*calendar.monthrange(year,month)[1]
        samples=range(sample_count)
        names=[str(station.get("name",f"站点{index+1}")) for index,station in enumerate(self.stations)]
        rates=[]
        for name,station in zip(names,self.stations):
            site={**station,"simulation_start_utc":self.epoch_iso,
                  "rain_grade":self.station_grades.get(name,"中雨")}
            rates.append([synthetic_surface_rain_rate_mmh(site,site["rain_grade"],first_s+index*1800.)
                          for index in samples])
        self.month_status.setText(f"{year} 年 {month:02d} 月 · 五站合成雨强 · 每 30 分钟采样（mm/h，未累加，未标定） · 点击曲线设置顶部仿真起始时间")
        self.month_chart.set_series(names,list(samples),rates,unit="mm/h",monthly=True)
        self.month_chart.series_start=month_start

    def _toggle_monthly(self,checked):
        self.month_toggle.setText("▼ 收起月度雨强（每 30 分钟）" if checked else "▶ 展开月度雨强（每 30 分钟）")
        self.month_panel.setVisible(checked)
        if checked:self._render_monthly()

    def refresh(self):
        if self.source.currentIndex()!=0:return
        self._update_time_context()
        self.chart.setMinimumHeight(190)
        start=self.start_min.value();end=max(start,self.end_min.value())
        step=(1,5,10)[self.interval.currentIndex()]
        minutes=list(range(start,end+1,step)) or [start]
        if minutes[-1]!=end:minutes.append(end)
        # Limit the preview width; the underlying model is evaluated at the
        # exact selected times, without changing the planning sample cadence.
        if len(minutes)>61:minutes=minutes[::max(1,math.ceil(len(minutes)/61))][:61]
        times=[f"{minute//60:02d}:{minute%60:02d}" for minute in minutes]
        stations=[{**station,"simulation_start_utc":self.epoch_iso} for station in self.stations]
        if self.history_mode:
            snapshot=self.forecast_snapshot or {}
            if not snapshot.get("stations"):
                self._clear("尚无覆盖仿真时段的历史降水数据")
                return
            if datetime.fromisoformat(snapshot["start_iso"])!=datetime.fromisoformat(self.epoch_iso):
                self._clear("历史降水数据与仿真起点不一致")
                return
            rates=[]
            epoch=datetime.fromisoformat(snapshot["start_iso"])
            first=snapshot["stations"][0]
            start_s=start*60.;end_s=end*60.
            cuts=sorted({start_s,end_s}|{(datetime.fromisoformat(value)-epoch).total_seconds()
                for value in first["times"] if start_s<(datetime.fromisoformat(value)-epoch).total_seconds()<end_s})
            if len(cuts)<2:cuts=[start_s,start_s+1800.]
            sample_seconds=cuts[:-1]
            self._chart_minutes=[value/60. for value in sample_seconds]
            times=[f"{int(a//3600):02d}:{int(a%3600//60):02d}–{int(b//3600):02d}:{int(b%3600//60):02d}" for a,b in zip(cuts,cuts[1:])]
            for station in stations:
                row=next((record for record in snapshot["stations"] if record.get("name")==station.get("name")),None)
                if row is None:
                    self._clear(f"缺少 {station.get('name','')} 的 IMERG 数据。")
                    return
                station["forecast_rain"]={"hour_ends_s":[(datetime.fromisoformat(value)-epoch).total_seconds() for value in row["times"]],"rates_mmh":row["rates"]}
                try:rates.append([forecast_rain_rate_mmh(station,sample_time) for sample_time in sample_seconds])
                except ValueError as exc:
                    self._clear(str(exc));return
            self._show(stations,times,rates,"IMERG 历史降水雨强 mm/h")
        else:
            rates=[[synthetic_surface_rain_rate_mmh({**station,"rain_grade":self.station_grades.get(station.get("name"),"轻雨")},"轻雨",minute*60)
                    for minute in minutes] for station in stations]
            self._chart_minutes=minutes
            self._show(stations,times,rates,"按日期合成的雨强（未标定）；偏低／基准／偏高为不同雨量情景")
            if stations and all(rate<=0. for row in rates for rate in row):
                self.notice.setText("当前所选时段在合成场景中没有降雨事件，因此各站雨强和雨衰均为 0。合成结果不是该日期的天气预测；如需指定日期的雨量，请选择“天气预报”并获取或导入数据。")
                self.notice.show()
        if self.month_panel.isVisible():self._render_monthly()

    def fetch_online(self):
        if not self.stations:
            self._clear("暂无信关站")
            return
        start=self.online_start.dateTime().toPython()
        if start.date()<datetime.now().date():
            self._clear("所选预报起点早于今天")
            return
        if QDateTime.currentDateTime().secsTo(self.online_start.dateTime())>15*86400:
            self._clear("所选预报起点超出接口可用范围（约 16 天）")
            return
        self._requested_epoch_iso=self.epoch_iso
        params={"latitude":",".join(str(s["lat"]) for s in self.stations),
                "longitude":",".join(str(s["lon"]) for s in self.stations),
                "hourly":"rain","timezone":"UTC",
                "start_date":datetime.fromisoformat(self.epoch_iso).date().isoformat(),
                "end_date":(datetime.fromisoformat(self.epoch_iso)+timedelta(seconds=self.duration_s+3600)).date().isoformat()}
        url="https://api.open-meteo.com/v1/forecast?"+urlencode(params)
        self.fetch_button.setEnabled(False);self._clear("正在获取逐小时预报…")
        self.forecast_snapshot=None
        threading.Thread(target=self._request_forecast,args=(url,),daemon=True).start()

    def _request_forecast(self,url):
        try:
            request=urllib.request.Request(url,headers={"User-Agent":"feeder-link-simulator/1.0"})
            with urllib.request.urlopen(request,timeout=20) as response:
                payload=json.load(response)
            self.online_result.emit(payload,None)
        except urllib.error.HTTPError as exc:
            detail=exc.read(300).decode("utf-8",errors="replace")
            self.online_result.emit(None,f"HTTP {exc.code}：{detail}")
        except (OSError,ValueError,TypeError) as exc:
            self.online_result.emit(None,f"{type(exc).__name__}：{exc}")

    def _forecast_received(self,payload,error):
        self.fetch_button.setEnabled(self.inputs_editable)
        if getattr(self,"_requested_epoch_iso",None)!=self.epoch_iso:
            self._clear("仿真起点已改变，请重新获取天气预报")
            return
        if error:
            self._clear(f"天气接口请求失败：{error}")
            return
        try:
            if isinstance(payload,dict) and payload.get("error"):
                raise ValueError(str(payload.get("reason") or payload))
            records=payload if isinstance(payload,list) else [payload]
            if len(records)!=len(self.stations):raise ValueError("返回的站点数与请求不一致")
            self.forecast_snapshot={"source":"Open-Meteo hourly forecast","start_iso":self.epoch_iso,
                "retrieved_at":datetime.now().isoformat(timespec="seconds"),"stations":[]}
            for station,record in zip(self.stations,records):
                hourly=record["hourly"]
                self.forecast_snapshot["stations"].append({"name":station.get("name"),
                    "lat":float(station["lat"]),"lon":float(station["lon"]),
                    "times":[value+"+00:00" for value in hourly["time"]],
                    "rates":[max(0.,float(value or 0)) for value in hourly["rain"]]})
            self._render_snapshot()
            if self.forecast_snapshot:self.forecast_loaded.emit(self.forecast_snapshot)
        except (ValueError,KeyError,TypeError,IndexError) as exc:
            self.forecast_snapshot=None
            self._clear(f"无法解析天气预报：{exc}")

    def import_forecast_csv(self):
        path,_=QFileDialog.getOpenFileName(self,"导入天气预报 CSV","","CSV 文件 (*.csv)")
        if not path:return
        try:
            with open(path,"r",encoding="utf-8-sig",newline="") as stream:
                rows=list(csv.DictReader(stream))
            required={"station","time_utc","rain_mm_h"}
            if not rows or not required.issubset(rows[0]):
                raise ValueError("CSV 需要 station,time_utc,rain_mm_h 三列")
            start=datetime.fromisoformat(self.epoch_iso)
            def utc_time(value):
                parsed=datetime.fromisoformat(value.replace("Z","+00:00"))
                return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
            records=[]
            for station in self.stations:
                name=station["name"]
                data=sorted((utc_time(row["time_utc"]),float(row["rain_mm_h"]))
                    for row in rows if row["station"]==name)
                if not data or data[0][0]>start+timedelta(hours=1) or data[-1][0]<start+timedelta(seconds=self.duration_s):
                    raise ValueError(f"{name} 的预报没有覆盖仿真时段")
                if any(value<0 for _,value in data):raise ValueError(f"{name} 的雨强不能为负")
                records.append({"name":name,"lat":station["lat"],"lon":station["lon"],
                    "times":[time.isoformat() for time,_ in data],"rates":[value for _,value in data]})
            self.forecast_snapshot={"source":"imported weather forecast CSV","start_iso":self.epoch_iso,
                "files":[path],"stations":records}
            self._render_snapshot();self.forecast_loaded.emit(self.forecast_snapshot)
        except (OSError,ValueError,TypeError) as exc:self._clear(f"天气预报导入失败：{exc}")
