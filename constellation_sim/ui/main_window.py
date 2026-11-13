#!/usr/bin/env python3
"""Qt desktop constellation simulator with an embedded Cesium 2D/3D view."""

from __future__ import annotations

import json
import csv
import math
import heapq
import sys
import tempfile
import threading
import time
import os
import shutil
import hashlib
import sqlite3
import marshal
import zlib
import gc
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import ephem

try:
    from PySide6.QtCore import QEasingCurve, QEvent, QObject, QPropertyAnimation, QSettings, QThread, QTimer, QUrl, Qt, Signal, Slot
    from PySide6.QtGui import QAction, QColor, QFont, QIcon, QPalette, QPainter, QPen, QPixmap
    from PySide6.QtWidgets import (
        QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
        QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
        QMessageBox, QPushButton, QProgressBar, QScrollArea, QSlider, QSpinBox, QSplitter,
        QStackedWidget, QStyle, QTableWidget, QTableWidgetItem, QTabBar, QTabWidget,
        QToolButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget, QSizePolicy,
        QHeaderView,
    )
except ImportError as exc:
    raise SystemExit("缺少 PySide6。请在启动软件的 Python 环境执行：python -m pip install --upgrade -r requirements.txt") from exc

try:
    from PySide6.QtWebEngineWidgets import QWebEngineView
    from PySide6.QtWebEngineCore import QWebEngineSettings
    from PySide6.QtWebChannel import QWebChannel
except ImportError as exc:
    raise SystemExit(
        "PySide6已安装，但Qt WebEngine运行库缺失或DLL版本不一致。\n"
        "请在独立的 Python 环境执行：python -m pip install --upgrade -r requirements.txt"
    ) from exc


MODULE_FILE = Path(__file__).resolve()
PROJECT_ROOT = MODULE_FILE.parents[2]
ROOT = next((root for root in (PROJECT_ROOT, MODULE_FILE.parents[3]) if (root / "satgenpy" / "satgen").is_dir()), PROJECT_ROOT)
WEB_FILE = Path(__file__).with_name("viewer.html")
DEFAULT_TRAFFIC_GRID = Path(__file__).with_name("worldpop_h3_res3.csv")
sys.path.insert(0, str(ROOT / "satgenpy"))
import satgen  # noqa: E402
from constellation_sim.algorithm.models import SimulationConfig
from constellation_sim.algorithm.result_store import ResultStore
from constellation_sim.ui.web_bridge import MapBridge
from constellation_sim.algorithm.traffic_model import TrafficCell, TrafficModel, TrafficModelConfig, calculate_visibility
from constellation_sim.algorithm.traffic_preprocess import load_cells_csv
from constellation_sim.algorithm.planning import SlowFastLoopPlanner, PlannerConfig, DEFAULT_MODCOD_TABLE, load_modcod_table, _graph, link_state, _slant_distance_km
from constellation_sim.algorithm.rain_model import scenario_attenuation_db, scenario_spatial_factor, install_scenario_spatial_cache, RAIN_GRADE_SPATIAL_PROFILES
from feeder_integration import create_calculation_pages, create_simulation_pages


EARTH_RADIUS_M = 6_378_135.0
EARTH_FLATTENING = 1.0 / 298.26
MU = 3.986004418e14
DIAGNOSTIC_LOG_LOCK = threading.Lock()


def file_content_hash(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()


def append_diagnostic(path, event, **fields):
    """Append one compact diagnostic event without involving the UI thread."""
    if not path:
        return
    record={"wall_time":datetime.now().astimezone().isoformat(),"event":event,**fields}
    try:
        with DIAGNOSTIC_LOG_LOCK:
            with Path(path).open("a",encoding="utf-8") as stream:
                stream.write(json.dumps(record,ensure_ascii=False,separators=(",",":"))+"\n")
    except OSError:
        pass


class CenteredTabWidget(QWidget):
    """Compact segmented navigation centered independently of page width."""
    currentChanged=Signal(int)

    def __init__(self,parent=None):
        super().__init__(parent);self._buttons=[];self._overlay_navigation=[]
        layout=QVBoxLayout(self);layout.setContentsMargins(0,0,0,0);layout.setSpacing(0);self._root_layout=layout
        navigation=QWidget();navigation.setObjectName("centeredTabNavigation");self._navigation=navigation
        nav_layout=QHBoxLayout(navigation);nav_layout.setContentsMargins(8,2,0,2);nav_layout.setSpacing(0);self._nav_layout=nav_layout
        self._segments=QWidget();self._segments.setObjectName("segmentedTabGroup")
        self._segment_layout=QHBoxLayout(self._segments);self._segment_layout.setContentsMargins(0,0,0,0);self._segment_layout.setSpacing(0)
        nav_layout.addWidget(self._segments,0,Qt.AlignmentFlag.AlignLeft);nav_layout.addStretch(1)
        self._stack=QStackedWidget();self._group=QButtonGroup(self);self._group.setExclusive(True)
        layout.addWidget(navigation);layout.addWidget(self._stack,1)

    def addNavigationWidget(self,widget):
        widget.setParent(self);widget.show();widget.raise_();self._overlay_navigation.append(widget)
        QTimer.singleShot(0,self._position_overlay_navigation)

    def _position_overlay_navigation(self):
        # A hidden navigation widget can retain its old expanded geometry.
        # Using that stale height placed the planning switcher in the middle
        # of the map after the outer workspace tabs were removed.
        y=(self._navigation.height() if self._navigation.isVisible() else 0)+8
        for widget in self._overlay_navigation:
            if not widget.isVisible():continue
            size=widget.sizeHint();widget.resize(size);widget.move(max(0,(self.width()-widget.width())//2),y);widget.raise_()
            y+=widget.height()+4

    def resizeEvent(self,event):
        super().resizeEvent(event);self._position_overlay_navigation()
        overlay=getattr(self,"busy_overlay",None)
        if overlay is not None:overlay.setGeometry(self.rect())

    def setNavigationVisible(self,visible):self._navigation.setVisible(visible)

    def segmentWidget(self):return self._segments

    def addTab(self,widget,title):
        index=self._stack.addWidget(widget);button=QPushButton(title);button.setObjectName("segmentedTabButton");button.setCheckable(True)
        button.clicked.connect(lambda checked,i=index:self.setCurrentIndex(i));self._group.addButton(button,index)
        self._segment_layout.addWidget(button);self._buttons.append(button)
        if index==0:button.setChecked(True)
        return index

    def currentIndex(self):return self._stack.currentIndex()

    def setCurrentIndex(self,index):
        if not 0<=index<self._stack.count():return
        changed=index!=self._stack.currentIndex();self._stack.setCurrentIndex(index);self._buttons[index].setChecked(True)
        if changed:self.currentChanged.emit(index)


class SeekSlider(QSlider):
    """Timeline slider that can be dragged from any point on the groove."""
    def _set_from_pointer(self,event):
        ratio=max(0.0,min(1.0,float(event.position().x())/max(1,self.width()-1)))
        value=self.minimum()+round(ratio*(self.maximum()-self.minimum()))
        if value!=self.value():self.setValue(value)
        self.sliderMoved.emit(value)
    def mousePressEvent(self,event):
        if event.button()==Qt.MouseButton.LeftButton:
            self.setSliderDown(True);self._set_from_pointer(event);event.accept();return
        super().mousePressEvent(event)
    def mouseMoveEvent(self,event):
        if self.isSliderDown() and event.buttons()&Qt.MouseButton.LeftButton:
            self._set_from_pointer(event);event.accept();return
        super().mouseMoveEvent(event)
    def mouseReleaseEvent(self,event):
        if event.button()==Qt.MouseButton.LeftButton and self.isSliderDown():
            self._set_from_pointer(event);self.setSliderDown(False);event.accept();return
        super().mouseReleaseEvent(event)


class BusyOverlay(QWidget):
    """Modal workspace progress indicator for long calculations."""
    def __init__(self,parent=None):
        super().__init__(parent);self._text="正在计算…";self._value=0;self._phase=0;self._started_at=None
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground,False)
        self._timer=QTimer(self);self._timer.timeout.connect(self._tick);self.hide()
    def start(self,text):
        self._text=str(text);self._value=0;self._phase=0;self._started_at=time.perf_counter()
        if self.parentWidget() is not None:self.setGeometry(self.parentWidget().rect())
        self.show();self.raise_();self._timer.start(80);self.update()
    def set_progress(self,value,text=None):
        self._value=max(0,min(100,int(value)))
        if text:self._text=str(text)
        if not self.isVisible():self.start(self._text);self._value=max(0,min(100,int(value)))
        self.raise_();self.update()
    def stop(self):
        self._timer.stop();self.hide();self._started_at=None
    def _tick(self):self._phase=(self._phase+1)%12;self.update()
    def paintEvent(self,event):
        painter=QPainter(self);painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(),QColor(2,13,21,185))
        cx=self.width()//2;cy=self.height()//2;card_w=min(520,max(340,self.width()-80));card_h=174
        card=self.rect().adjusted(cx-card_w//2,cy-card_h//2,-(self.width()-cx-card_w//2),-(self.height()-cy-card_h//2))
        painter.setPen(QPen(QColor("#17647d"),1));painter.setBrush(QColor(7,31,45,245));painter.drawRoundedRect(card,10,10)
        painter.save();painter.translate(cx,cy-34);painter.rotate(self._phase*30)
        painter.setPen(QPen(QColor("#35e6ff"),5,Qt.PenStyle.SolidLine,Qt.PenCapStyle.RoundCap));painter.drawArc(-22,-22,44,44,20*16,270*16);painter.restore()
        painter.setPen(QColor("#e7fbff"));font=painter.font();font.setPointSize(11);font.setBold(True);painter.setFont(font)
        painter.drawText(card.adjusted(22,78,-22,-52),Qt.AlignmentFlag.AlignHCenter|Qt.AlignmentFlag.AlignTop,self._text)
        elapsed=max(0,int(time.perf_counter()-self._started_at)) if self._started_at is not None else 0
        painter.setPen(QColor("#8dc7d8"));font.setPointSize(9);font.setBold(False);painter.setFont(font)
        painter.drawText(card.adjusted(22,112,-22,-18),Qt.AlignmentFlag.AlignHCenter|Qt.AlignmentFlag.AlignTop,f"进度 {self._value}%　已用时 {elapsed//60:02d}:{elapsed%60:02d}")


class ResultTrendChart(QWidget):
    """One-unit time chart: delivered business or propagation delay."""
    def __init__(self,kind="business"):
        super().__init__();self.kind=kind;self.rows=[];self.hover_index=None
        self.setMinimumHeight(200);self.setMouseTracking(True)
    def append_values(self,time_s,*values):
        row=[]
        for value in (time_s,*values):
            try:number=float(value)
            except (TypeError,ValueError,OverflowError):number=0.
            row.append(number if math.isfinite(number) else 0.)
        row=tuple(row)
        if self.rows and row[0]<self.rows[-1][0]:self.rows.clear()
        if self.rows and row[0]==self.rows[-1][0]:self.rows[-1]=row
        else:self.rows.append(row)
        self.rows=self.rows[-600:];self.update()
    def clear(self):self.rows.clear();self.hover_index=None;self.update()
    def mouseMoveEvent(self,event):
        if len(self.rows)<2:return
        left,right=64,24;w=max(1,self.width()-left-right);x=event.position().x()
        if not left<=x<=left+w:self.hover_index=None
        else:
            t0,t1=self.rows[0][0],self.rows[-1][0];target=t0+(x-left)/w*max(t1-t0,1e-9)
            self.hover_index=min(range(len(self.rows)),key=lambda i:abs(self.rows[i][0]-target))
        self.update()
    def leaveEvent(self,_event):self.hover_index=None;self.update()
    def paintEvent(self,_event):
        p=QPainter(self);p.setRenderHint(QPainter.RenderHint.Antialiasing);p.fillRect(self.rect(),QColor("#071825"))
        left,right,top,bottom=64,24,40,34;w=max(1,self.width()-left-right);h=max(1,self.height()-top-bottom)
        p.setPen(QPen(QColor("#315164"),1));p.drawRect(left,top,w,h)
        if len(self.rows)<2:
            p.setPen(QColor("#8fb3c5"));p.drawText(self.rect(),Qt.AlignmentFlag.AlignCenter,"开始仿真后显示趋势");p.end();return
        names=("落地需求","已送达业务","可用链路容量") if self.kind=="business" else ("加权平均时延","P95 时延")
        colors=("#a8c4d4","#32e28b","#24d7ff") if self.kind=="business" else ("#ffd45a","#ff6a92")
        rows=self.rows;t0,t1=rows[0][0],rows[-1][0];span=max(t1-t0,1e-6)
        scale=max((max(row[1:]) for row in rows),default=1.) or 1.
        p.setFont(QFont("Microsoft YaHei",8))
        for i in range(5):
            ratio=i/4;y=top+h-int(h*ratio)
            p.setPen(QPen(QColor("#173747"),1));p.drawLine(left,y,left+w,y)
            p.setPen(QColor("#84acbd"));p.drawText(2,y-9,56,18,Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignVCenter,f"{scale*ratio:.1f}")
            x=left+int(w*ratio);p.drawText(x-28,top+h+5,56,18,Qt.AlignmentFlag.AlignCenter,f"{t0+span*ratio:.0f}s")
        unit="Gb/s" if self.kind=="business" else "ms"
        p.setPen(QColor("#a9d3e2"));p.drawText(5,7,60,20,Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter,unit)
        legend_x=left
        for col,(name,color) in enumerate(zip(names,colors),1):
            p.setPen(QPen(QColor(color),2));points=[]
            for row in rows:
                x=left+w*max(0.,min(1.,(row[0]-t0)/span));y=top+h-h*max(0.,min(1.,row[col]/scale));points.append((int(x),int(y)))
            for a,b in zip(points,points[1:]):p.drawLine(a[0],a[1],b[0],b[1])
            p.drawLine(legend_x,19,legend_x+16,19);p.setPen(QColor("#c8e5f2"));p.drawText(legend_x+22,7,100,24,Qt.AlignmentFlag.AlignVCenter,name);legend_x+=125
        if self.hover_index is not None:
            row=rows[self.hover_index];x=left+int(w*max(0.,min(1.,(row[0]-t0)/span)))
            p.setPen(QPen(QColor("#d9f5ff"),1,Qt.PenStyle.DashLine));p.drawLine(x,top,x,top+h)
            label=f"t={row[0]:.1f}s　"+"　".join(f"{name} {value:.3f} {unit}" for name,value in zip(names,row[1:]))
            width=min(max(200,self.width()-20),650);bx=max(10,min(x-width//2,self.width()-width-10))
            p.fillRect(bx,top+8,width,25,QColor("#0b3042"));p.setPen(QColor("#e7fbff"));p.drawText(bx+8,top+8,width-16,25,Qt.AlignmentFlag.AlignVCenter,label)
        p.end()


class GatewayScopeTabs(QTabBar):
    """Scrollable station tabs with the small data API used by result pages."""
    def __init__(self,parent=None):
        super().__init__(parent);self.setExpanding(False);self.setUsesScrollButtons(True);self.setElideMode(Qt.TextElideMode.ElideRight)
        self.setDrawBase(False);self.setMaximumHeight(34)
        self.setStyleSheet("QTabBar::tab{background:#0a2636;color:#a9cfdd;border:1px solid #174b62;border-right:0;padding:7px 16px;min-width:72px} QTabBar::tab:first{border-top-left-radius:5px;border-bottom-left-radius:5px} QTabBar::tab:last{border-right:1px solid #174b62;border-top-right-radius:5px;border-bottom-right-radius:5px} QTabBar::tab:selected{background:#08718a;color:white;border-color:#28d9f5;font-weight:700} QTabBar::tab:hover:!selected{background:#0d3a4d;color:#d8f7ff}")
    def addItem(self,text,data=None):
        index=self.addTab(text);self.setTabData(index,data);return index
    def itemData(self,index):return self.tabData(index)
    def currentData(self):return self.tabData(self.currentIndex()) if self.currentIndex()>=0 else None
    def clear(self):
        while self.count():self.removeTab(0)
    def findData(self,data):
        for index in range(self.count()):
            if self.tabData(index)==data:return index
        return -1


def _gateway_scope_row(combo):
    row=QWidget();layout=QHBoxLayout(row);layout.setContentsMargins(0,0,0,2)
    layout.addWidget(combo,1)
    return row


def _rain_scope_row(selector):
    row=QWidget();layout=QHBoxLayout(row);layout.setContentsMargins(0,0,0,2)
    layout.addWidget(selector,1)
    return row


def _rain_severity_text(value):
    return {"clear":"无明显雨衰","light":"轻度","moderate":"中度","heavy":"重度"}.get(str(value).lower(),str(value or "--"))


def _export_table_csv(parent,table,default_name):
    if table.rowCount()<=0:
        QMessageBox.information(parent,"导出表格","当前表格没有可导出的数据。")
        return
    suggested=f"{default_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    path,_=QFileDialog.getSaveFileName(parent,"导出表格",suggested,"CSV表格 (*.csv)")
    if not path:return
    if not path.lower().endswith(".csv"):path+=".csv"
    columns=[column for column in range(table.columnCount()) if not table.isColumnHidden(column)]
    try:
        with open(path,"w",encoding="utf-8-sig",newline="") as stream:
            writer=csv.writer(stream)
            writer.writerow([(table.horizontalHeaderItem(column).text() if table.horizontalHeaderItem(column) else f"第{column+1}列") for column in columns])
            for row in range(table.rowCount()):
                writer.writerow([(table.item(row,column).text() if table.item(row,column) else "") for column in columns])
        QMessageBox.information(parent,"导出完成",f"表格已保存到：\n{path}")
    except OSError as exc:
        QMessageBox.critical(parent,"导出失败",str(exc))


def _table_export_bar(parent,table,default_name):
    bar=QWidget();layout=QHBoxLayout(bar);layout.setContentsMargins(0,0,0,2);layout.addStretch(1)
    button=QPushButton("导出表格");button.setToolTip("将当前表格的中文表头和全部数据导出为CSV文件")
    button.clicked.connect(lambda _checked=False:_export_table_csv(parent,table,default_name));layout.addWidget(button)
    return bar

def _sync_gateway_scope(combo,station_names):
    current=combo.currentData();names=list(station_names or [])
    expected=[None]+list(range(len(names)))
    if combo.count()==len(expected) and [combo.itemData(i) for i in range(combo.count())]==expected:return
    combo.blockSignals(True);combo.clear();combo.addItem("全网汇总",None)
    for gid,name in enumerate(names):combo.addItem(str(name),gid)
    index=combo.findData(current);combo.setCurrentIndex(index if index>=0 else 0);combo.blockSignals(False)


def _gateway_name(station_names,gid):
    return station_names[gid] if 0<=gid<len(station_names) else f"GW-{gid+1:02d}"


def _station_metrics(data,gid):
    planning=data.get("planning") or {};plan=planning.get("active_plan") or {};active=planning.get("active_evaluation") or {}
    planned=[x for x in plan.get("links",[]) if gid is None or int(x.get("gateway",-1))==gid]
    actual=[x for x in active.get("active_links",[]) if gid is None or int(x.get("gateway",-1))==gid]
    planned_feeders={int(x.get("satellite",-1)) for x in planned};actual_feeders={int(x.get("satellite",-1)) for x in actual}
    plan_routes=[r for r in plan.get("routes",[]) if gid is None or int(r.get("feeder",-1)) in planned_feeders]
    execution=planning.get("execution_plan") or plan
    evaluated_routes=active.get("routes")
    route_source=evaluated_routes if evaluated_routes is not None else execution.get("routes",[])
    actual_routes=[r for r in route_source if gid is None or int(r.get("feeder",-1)) in actual_feeders]
    ideal=sum(float(x.get("rate_gbps",0)) for x in planned);capacity=sum(float(x.get("rate_gbps",0)) for x in actual)
    scheduled=sum(float(x.get("scheduled_gbps",0)) for x in actual);demand=sum(float(r.get("demand_gbps",0)) for r in plan_routes)
    rain=[float(x.get("rain_actual_db",0)) for x in actual]
    weighted=sum(float(r.get("hops",0))*float(r.get("demand_gbps",0)) for r in actual_routes)
    route_demand=sum(float(r.get("demand_gbps",0)) for r in actual_routes)
    delays=sorted((float(r.get("propagation_delay_ms",float(r.get("distance_km",0))/299.792458)),float(r.get("demand_gbps",0))) for r in actual_routes)
    average_delay=sum(delay*demand_value for delay,demand_value in delays)/max(route_demand,1e-12) if delays else 0.;p95_delay=0.;cumulative=0.
    for delay,demand_value in delays:
        cumulative+=demand_value
        if cumulative>=.95*route_demand:p95_delay=delay;break
    planned_delays=sorted((float(r.get("propagation_delay_ms",float(r.get("distance_km",0))/299.792458)),float(r.get("demand_gbps",0))) for r in plan_routes)
    planned_demand=sum(x[1] for x in planned_delays);planned_average_delay=sum(x[0]*x[1] for x in planned_delays)/max(planned_demand,1e-12) if planned_delays else 0.;planned_p95_delay=0.;cumulative=0.
    for delay,demand_value in planned_delays:
        cumulative+=demand_value
        if cumulative>=.95*planned_demand:planned_p95_delay=delay;break
    planned_average_hops=sum(float(r.get("hops",0)) for r in plan_routes)/len(plan_routes) if plan_routes else 0.
    planned_weighted_hops=sum(float(r.get("hops",0))*float(r.get("demand_gbps",0)) for r in plan_routes)/max(planned_demand,1e-12) if plan_routes else 0.
    return {"planned":planned,"actual":actual,"plan_routes":plan_routes,"actual_routes":actual_routes,
            "ideal":ideal,"capacity":capacity,"scheduled":scheduled,"demand":demand,
            "capacity_ratio":capacity/max(ideal,1e-12) if planned else 0.,
            "route_ratio":len(actual_routes)/max(len(plan_routes),1) if plan_routes else 1.,
            "rain_average":sum(rain)/len(rain) if rain else 0.,"rain_max":max(rain) if rain else 0.,
            "average_delay_ms":average_delay,"p95_delay_ms":p95_delay,"maximum_delay_ms":max((x[0] for x in delays),default=0.),
            "planned_average_delay_ms":planned_average_delay,"planned_p95_delay_ms":planned_p95_delay,"planned_maximum_delay_ms":max((x[0] for x in planned_delays),default=0.),
            "planned_average_hops":planned_average_hops,"planned_weighted_hops":planned_weighted_hops,
            "average_hops":sum(float(r.get("hops",0)) for r in actual_routes)/len(actual_routes) if actual_routes else 0.,
            "weighted_hops":weighted/max(route_demand,1e-12) if actual_routes else 0.}


def _delivery_gap_reason(active,demand,delivered,route_ratio):
    gap=max(0.,float(demand)-float(delivered))
    if gap<=.001:return "需求已满足"
    deliverable=float((active or {}).get("theoretical_deliverable_gbps",delivered))
    link_gap=max(0.,float(demand)-deliverable)
    allocation_gap=max(0.,float((active or {}).get("route_capacity_gap_gbps",deliverable-delivered)))
    parts=[]
    if route_ratio<.999:parts.append(f"无有效路径卫星占比 {100*(1-route_ratio):.1f}%")
    if link_gap>.001:parts.append(f"可用馈电容量缺口 {link_gap:.3f} Gb/s")
    if allocation_gap>.001:parts.append(f"路径容量分配缺口 {allocation_gap:.3f} Gb/s")
    return "；".join(parts) if parts else f"业务分配缺口 {gap:.3f} Gb/s"


class PlanningResultsWidget(QWidget):
    """Show slow-loop planning generations and the active resource plan."""
    def __init__(self):
        super().__init__();layout=QVBoxLayout(self);layout.setContentsMargins(12,48,12,12);layout.setSpacing(8)
        self.gateway_scope=GatewayScopeTabs();self.gateway_scope.currentChanged.connect(lambda _index:self._scope_changed())
        self.latest_data=None;self.station_names=[];self.deployment_result={};self.plan_history=[]
        self.offline_catalog=[];self.offline_visible_indices=[];self.selected_offline_index=None
        self.rain_scope=GatewayScopeTabs()
        for title,value in (("全部工况",None),("轻雨","轻雨"),("中雨","中雨"),("重雨","重雨")):self.rain_scope.addItem(title,value)
        self.rain_scope.currentChanged.connect(lambda _index:self._render_offline_catalog())
        self.summary=QLabel("慢环规划将在开始仿真后生成");self.summary.setObjectName("planningKpi");self.summary.setWordWrap(True)
        self.plans=QTableWidget(0,11);self.plans.setHorizontalHeaderLabels(["生效时段","雨衰工况","使用信关站","馈电链路","链路总容量 Gb/s","落地需求 Gb/s","规划可落地 Gb/s","规划满足率","平均时延 ms","P95时延 ms","校核结果"])
        self.plan_links=QTableWidget(0,9);self.plan_links.setHorizontalHeaderLabels(["信关站","馈电卫星","最低仰角 °","雨衰 dB","Es/N0 dB","调制编码","容量 Gb/s","雨衰等级","卫星 TLE"])
        self.plan_routes=QTableWidget(0,6);self.plan_routes.setHorizontalHeaderLabels(["业务卫星","规划馈电卫星","ISL跳数","路径距离 km","业务 Gb/s","传播时延 ms"])
        for table in (self.plans,self.plan_links,self.plan_routes):
            self._style_table(table)
        link_header=self.plan_links.horizontalHeader();link_header.setStretchLastSection(False)
        for column in range(7):link_header.setSectionResizeMode(column,QHeaderView.ResizeMode.Stretch)
        link_header.setSectionResizeMode(7,QHeaderView.ResizeMode.Fixed);self.plan_links.setColumnWidth(7,96)
        link_header.setSectionResizeMode(8,QHeaderView.ResizeMode.Fixed);self.plan_links.setColumnWidth(8,100)
        self.plan_links.horizontalHeaderItem(2).setToolTip("该五分钟规划时段内，6个采样点的最低仰角")
        self.plan_links.horizontalHeaderItem(3).setToolTip("规划计算采用的雨衰值，单位 dB")
        self.plans.cellClicked.connect(lambda row,_column:self._show_offline_plan_links(row))
        self.plan_links.cellClicked.connect(self._show_tle_cell)
        self.slow_title=QLabel("慢周期站星规划");self.slow_title.setStyleSheet("font-size:14px;font-weight:700;color:#63e8ff")
        self.plans_label=QLabel("规划批次");self.links_label=QLabel("规划馈电链路");self.routes_label=QLabel("逐跳路径用于内部可达性校核；请在地图点击卫星按需查看")
        layout.addWidget(_gateway_scope_row(self.gateway_scope));layout.addWidget(_rain_scope_row(self.rain_scope))
        self.result_tabs=QTabWidget();self.result_tabs.setDocumentMode(True);self.result_tabs.setObjectName("planningResultTabs")
        batch_page=QWidget();batch_layout=QVBoxLayout(batch_page);batch_layout.setContentsMargins(8,8,8,8);batch_layout.setSpacing(7)
        batch_layout.addWidget(self.slow_title);batch_layout.addWidget(self.summary);batch_layout.addWidget(self.plans_label);batch_layout.addWidget(_table_export_bar(self,self.plans,"规划批次"));batch_layout.addWidget(self.plans,1)
        link_page=QWidget();link_layout=QVBoxLayout(link_page);link_layout.setContentsMargins(8,8,8,8);link_layout.setSpacing(7)
        self.time_scope=GatewayScopeTabs()
        for minute in range(0,30,5):self.time_scope.addItem(f"{minute:02d}:00",minute*60)
        self.time_scope.currentChanged.connect(lambda _index:self._time_scope_changed())
        link_layout.addWidget(self.time_scope)
        link_layout.addWidget(self.links_label);link_layout.addWidget(_table_export_bar(self,self.plan_links,"规划馈电链路"));link_layout.addWidget(self.plan_links,1);link_layout.addWidget(self.routes_label)
        self.result_tabs.addTab(batch_page,"规划批次");self.result_tabs.addTab(link_page,"馈电链路")
        layout.addWidget(self.result_tabs,1)
        self.slow_widgets=(self.slow_title,self.summary,self.plans_label,self.plans,self.links_label,self.plan_links,self.routes_label)
        self.plan_routes.hide()
        self.time_scope.hide()
        for widget in self.slow_widgets:widget.hide()
    @staticmethod
    def _style_table(table):
            table.setAlternatingRowColors(True);table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows);table.setShowGrid(False);table.verticalHeader().setVisible(False)
            table.verticalHeader().setDefaultSectionSize(28);table.horizontalHeader().setMinimumSectionSize(74);table.horizontalHeader().setDefaultSectionSize(118)
            # ResizeToContents rescanned every cell whenever live tables were
            # refreshed, which visibly blocked the GUI on link-heavy frames.
            table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive);table.horizontalHeader().setStretchLastSection(True)
            table.setStyleSheet("QTableWidget{border:1px solid #173e52;alternate-background-color:#0a2231;selection-background-color:#10566f} QHeaderView::section{background:#0d3043;color:#74e9ff;border:0;border-right:1px solid #1a5068;padding:7px 9px;font-weight:600} QTableWidget::item{padding:4px 8px;border-bottom:1px solid #102f40}")
    @staticmethod
    def _fill(table,row,values):
        table.insertRow(row)
        for col,value in enumerate(values):
            item=QTableWidgetItem(str(value));item.setTextAlignment(Qt.AlignmentFlag.AlignVCenter|Qt.AlignmentFlag.AlignHCenter);table.setItem(row,col,item)
    def update_frame(self,data,station_names):
        self.latest_data=data;self.station_names=list(station_names);_sync_gateway_scope(self.gateway_scope,self.station_names)
        for widget in self.slow_widgets:widget.show()
        planning=data.get("planning") or {};plans=planning.get("plans",[])
        if planning.get("planning_event"):
          event_key=(int(data.get("index",-1)),int(planning.get("generation",0)))
          previous_key=(int(self.plan_history[-1][0].get("index",-2)),int((self.plan_history[-1][0].get("planning") or {}).get("generation",-1))) if self.plan_history else None
          if event_key!=previous_key:self.plan_history.append((data, list(station_names)));self.plan_history=self.plan_history[-200:]
        self._render()
    def _scope_changed(self):
        self._render()
        if self.offline_catalog and self.selected_offline_index is not None:self._show_offline_plan_index(self.selected_offline_index)
    def _render(self):
        if not self.latest_data:return
        data=self.latest_data;station_names=self.station_names;gid=self.gateway_scope.currentData();planning=data.get("planning") or {};plan=planning.get("active_plan") or {}
        metrics=_station_metrics(data,gid);scope="全网" if gid is None else _gateway_name(station_names,gid)
        plan=planning.get("active_plan") or {};demand=float(plan.get("total_demand_gbps",0))
        scope_demand=demand if gid is None else sum(float(r.get("demand_gbps",0)) for r in metrics["plan_routes"])
        self.summary.setText(f"{scope} · 第 {planning.get('generation',0)} 批滚动规划　生成 t={planning.get('generated_at_s',0):.2f}s　下次规划 t={planning.get('next_planning_time_s',0):.2f}s\n预测窗口 {planning.get('planning_horizon_s',0):g}s　时间片 {planning.get('planning_slot_s',0):g}s\n规划馈电链路 {len(metrics['planned'])} 条　规划容量 {metrics['ideal']:.3f} Gb/s　经该站出口的可达业务 {scope_demand:.3f} Gb/s\n网络传播时延（服务卫星→信关站） 加权平均 {metrics['planned_average_delay_ms']:.3f} ms / P95 {metrics['planned_p95_delay_ms']:.3f} ms / 最大 {metrics['planned_maximum_delay_ms']:.3f} ms")
        if not self.offline_catalog:
            self.plans.setRowCount(0)
            history=self.plan_history or [(data,station_names)]
            for event,event_names in history:
                ep=event.get("planning") or {};em=_station_metrics(event,gid);eplan=ep.get("active_plan") or {}
                if gid is not None and not em["planned"]:continue
                usage="；".join(f"{_gateway_name(event_names,int(x['gateway']))} {x['used']}/{x['available']}" for x in eplan.get("station_usage",[]) if gid is None or int(x.get("gateway",-1))==gid)
                delivered=sum(float(r.get('demand_gbps',0)) for r in em['plan_routes']);demand=float(eplan.get('total_demand_gbps',0));ratio=100*delivered/max(demand,1e-12)
                self._fill(self.plans,self.plans.rowCount(),[f"t={event['time_s']:.2f}s",f"第{ep.get('generation',0)}次",eplan.get("gateway_count",0),len(em["planned"]),f"{em['ideal']:.3f}",f"{demand:.3f}",f"{delivered:.3f}",f"{ratio:.1f}%",f"{em['planned_average_delay_ms']:.3f}",f"{em['planned_p95_delay_ms']:.3f}","执行中"])
        if not self.offline_catalog:
            self.plan_links.setRowCount(0)
            for i,x in enumerate(metrics["planned"]):
                gid=int(x.get("gateway",0));name=station_names[gid] if gid<len(station_names) else f"GW-{gid+1:02d}"
                self._fill(self.plan_links,i,[name,f"SAT-{int(x.get('satellite',0))+1:05d}",f"{x.get('elevation_deg',0):.2f}°",f"{x.get('rain_attenuation_db',0):.2f}",f"{x.get('snr_db',0):.2f}",x.get("modcod","--"),f"{x.get('rate_gbps',0):.3f}",_rain_severity_text(x.get("rain_severity","--"))])
        self._render_deployment()
    def set_gateway_deployment(self,result):
        self.deployment_result=result or {};self._render_deployment()
    def _show_tle_cell(self,row,column):
        if column!=8:return
        item=self.plan_links.item(row,column)
        tle=item.data(Qt.ItemDataRole.UserRole) if item else None
        if tle:QMessageBox.information(self,"卫星 TLE",f"{tle['name']}\n{tle['line1']}\n{tle['line2']}")
    def set_offline_catalog(self,catalog):
        self.offline_catalog=list(catalog or []);self.selected_offline_index=None;self._render_offline_catalog()
        self.time_scope.setVisible(bool(self.offline_catalog))
        total_ms=float(self.offline_catalog[0].get("batch_total_calculation_ms",0)) if self.offline_catalog else 0
        source="基础数据缓存复用" if self.offline_catalog and self.offline_catalog[0].get("planning_cache_reused") else "本次重新计算"
        average_ms=sum(float(row.get("calculation_ms",0)) for row in self.offline_catalog)/max(1,len(self.offline_catalog))
        self.summary.setText(f"离线规划完成：3种雨衰工况 × 6个五分钟时段，共 {len(self.offline_catalog)} 个方案。\n来源：{source}　批次总耗时 {total_ms/1000:.2f} s　单方案平均 {average_ms/1000:.2f} s。仿真只执行所选方案，不触发紧急规划。")
        if self.offline_catalog:
            first=self.offline_catalog[0]
            mode=("容量时延加权策略" if first.get("selection_strategy")=="window_margin"
                  else "容量可行优先时延策略")
            self.summary.setText(self.summary.text()+f"\n{mode}")
        for widget in self.slow_widgets:widget.show()
    def _render_offline_catalog(self):
        if not self.offline_catalog:return
        grade=self.rain_scope.currentData();self.plans.setRowCount(0)
        self.offline_visible_indices=[index for index,plan in enumerate(self.offline_catalog) if grade is None or plan.get("rain_grade")==grade]
        for index in self.offline_visible_indices:
            plan=self.offline_catalog[index]
            self._fill(self.plans,self.plans.rowCount(),[
                f"{int(plan.get('valid_from_s',0)//60):02d}:00",plan.get("rain_grade","--"),plan.get("gateway_count",0),len(plan.get("links",[])),
                f"{float(plan.get('total_capacity_gbps',0)):.3f}",f"{float(plan.get('total_demand_gbps',0)):.3f}",f"{float(plan.get('reachable_demand_gbps',0)):.3f}",
                f"{100*float(plan.get('planned_delivery_ratio',0)):.1f}%",f"{float(plan.get('business_propagation_delay_ms',0)):.3f}",f"{float(plan.get('p95_propagation_delay_ms',0)):.3f}",
                ("起点达99%（已筛全段可见）" if plan.get("delivery_target_satisfied") and plan.get("full_window_visibility_required") else ("达到99%目标" if plan.get("delivery_target_satisfied") else f"缺口 {float(plan.get('delivery_gap_gbps',0)):.3f} Gb/s"))])
        if self.offline_visible_indices:
            selected=self.selected_offline_index if self.selected_offline_index in self.offline_visible_indices else next((index for index in self.offline_visible_indices if int(float(self.offline_catalog[index].get("valid_from_s",-1)))==self.time_scope.currentData()),self.offline_visible_indices[0])
            row=self.offline_visible_indices.index(selected);self.plans.selectRow(row);self._show_offline_plan_links(row)
    def _show_offline_plan_links(self,row):
        if not (0<=int(row)<len(self.offline_visible_indices)):return
        self._show_offline_plan_index(self.offline_visible_indices[int(row)])
    def _time_scope_changed(self):
        if not self.offline_catalog:return
        time_s=self.time_scope.currentData();grade=self.rain_scope.currentData()
        index=next((i for i,plan in enumerate(self.offline_catalog)
                    if int(float(plan.get("valid_from_s",-1)))==time_s and (grade is None or plan.get("rain_grade")==grade)),None)
        if index is not None:
            if index in self.offline_visible_indices:self.plans.selectRow(self.offline_visible_indices.index(index))
            self._show_offline_plan_index(index)
    def _show_offline_plan_index(self,index):
        if not (0<=int(index)<len(self.offline_catalog)):return
        self.selected_offline_index=int(index);plan=self.offline_catalog[self.selected_offline_index];self.plan_links.setRowCount(0)
        time_index=self.time_scope.findData(int(float(plan.get("valid_from_s",0))))
        if time_index>=0:
            self.time_scope.blockSignals(True);self.time_scope.setCurrentIndex(time_index);self.time_scope.blockSignals(False)
        gateway_filter=self.gateway_scope.currentData()
        links=[link for link in plan.get("links",[]) if gateway_filter is None or int(link.get("gateway",-1))==gateway_filter]
        links.sort(key=lambda link:(int(link.get("gateway",-1)),int(link.get("satellite",-1))))
        for row,link in enumerate(links):
            gateway=int(link.get("gateway",0));name=self.station_names[gateway] if gateway<len(self.station_names) else f"GW-{gateway+1:02d}"
            self._fill(self.plan_links,row,[name,f"SAT-{int(link.get('satellite',0))+1:05d}",f"{float(link.get('elevation_deg',0)):.2f}°",
                f"{float(link.get('rain_attenuation_db',0)):.2f}",f"{float(link.get('snr_db',0)):.2f}",link.get("modcod","--"),
                f"{float(link.get('rate_gbps',0)):.3f}",_rain_severity_text(link.get("rain_severity",plan.get("rain_grade","--"))),"查看 TLE" if link.get("tle") else "—"])
            self.plan_links.item(row,8).setData(Qt.ItemDataRole.UserRole,link.get("tle"))
        scope="全网" if gateway_filter is None else _gateway_name(self.station_names,gateway_filter)
        self.links_label.setText(f"规划馈电链路 · {int(float(plan.get('valid_from_s',0))//60):02d}:00 · {plan.get('rain_grade','--')} · {scope} · {len(links)} 条")
        if plan.get("selection_strategy")=="window_margin":
            spare=sum(float(link.get("planned_window_spare_gbps",0.)) for link in links)
            self.links_label.setText(self.links_label.text()+f" | \u9884\u8ba1\u5168\u65f6\u6bb5\u5269\u4f59 {spare:.3f} Gb/s")
    def _render_deployment(self):
        pass  # Deployment data remains available to the backend; its page was removed.
    def clear(self):
        for table in (self.plans,self.plan_links,self.plan_routes):table.setRowCount(0)
        self.time_scope.hide()
        self.summary.setText("慢环规划将在开始仿真后生成")
        self.latest_data=None;self.plan_history=[];self.offline_catalog=[];self.offline_visible_indices=[];self.selected_offline_index=None
        for widget in self.slow_widgets:widget.hide()


class PlanningResourceWidget(QWidget):
    """Fast-loop execution, including actual resources, links and routes."""
    def __init__(self):
        super().__init__();layout=QVBoxLayout(self);layout.setContentsMargins(12,48,12,12);layout.setSpacing(8)
        self.gateway_scope=GatewayScopeTabs();self.gateway_scope.currentChanged.connect(lambda _index:self._render(True,False))
        self.latest_data=None;self.station_names=[];self.frame_history=[]
        self.summary=QLabel("等待调度执行结果");self.summary.setObjectName("planningKpi");self.summary.setWordWrap(True);layout.addWidget(self.summary)
        layout.addWidget(_gateway_scope_row(self.gateway_scope))
        self.history=QTableWidget(0,11);self.history.setHorizontalHeaderLabels(["时刻","落地需求 Gb/s","实际送达 Gb/s","未送达 Gb/s","业务送达率","有路业务卫星比例","P95时延 ms","有效馈电链路","累计局部替代","调度结果","未送达原因"])
        self.deployment=QTableWidget(0,6);self.deployment.setHorizontalHeaderLabels(["信关站","可见卫星","天线占用","规划信道","建议天线","连接状态"])
        self.links=QTableWidget(0,9);self.links.setHorizontalHeaderLabels(["信关站","馈电卫星","仰角","雨衰 dB","Es/N0 dB","可用容量 Gb/s","已用容量 Gb/s","剩余容量 Gb/s","执行状态"])
        self.routes=QTableWidget(0,7);self.routes.setHorizontalHeaderLabels(["业务卫星","馈电卫星","ISL跳数","路径距离 km","业务 Gb/s","传播时延 ms","路径状态"])
        for table in (self.history,self.deployment,self.links,self.routes):PlanningResultsWidget._style_table(table)
        self.execution_tabs=QTabWidget();self.execution_tabs.setDocumentMode(True);self.execution_tabs.setObjectName("planningResultTabs")
        for title,table in (("执行时序",self.history),("信关站资源",self.deployment),("馈电链路",self.links)):
            page=QWidget();page_layout=QVBoxLayout(page);page_layout.setContentsMargins(8,8,8,8);page_layout.setSpacing(6);page_layout.addWidget(table,1);self.execution_tabs.addTab(page,title)
        layout.addWidget(self.execution_tabs,1)
        path_note=QLabel("ISL路径用于内部可达性与业务交付校核；在地图点击卫星可查看对应路径。")
        path_note.setStyleSheet("color:#8fb8c8;background:#081d2a;border-left:3px solid #28dfff;padding:7px 9px");layout.addWidget(path_note)
        self.routes.hide()
    def update_frame(self,data,station_names):
        self.latest_data=data;self.station_names=list(station_names);_sync_gateway_scope(self.gateway_scope,self.station_names)
        added=not self.frame_history or self.frame_history[-1][0].get("index")!=data.get("index")
        if added:
            self.frame_history.append((data,list(station_names)));self.frame_history=self.frame_history[-500:]
        self._render(False,added)
    def _render(self,rebuild_history=False,append_history=False):
        if not self.latest_data:return
        data=self.latest_data;station_names=self.station_names;gid=self.gateway_scope.currentData();scope="全网" if gid is None else _gateway_name(station_names,gid)
        planning=data.get("planning") or {};active=planning.get("active_evaluation") or {};deployment=planning.get("gateway_deployment") or {}
        metrics=_station_metrics(data,gid);delivered=float(planning.get("delivered_gbps",0)) if gid is None else metrics["scheduled"]
        unserved=float(active.get("unserved_gbps",0)) if gid is None else max(0,metrics["demand"]-delivered)
        route_ratio=float(active.get("route_available_ratio",0)) if gid is None else metrics["route_ratio"]
        demand=float((planning.get("decision_context") or {}).get("traffic_demand_gbps",0)) if gid is None else float(metrics["demand"])
        reason=_delivery_gap_reason(active,demand,delivered,route_ratio)
        self.summary.setText(f"{scope} · t={data['time_s']:.2f}s　需求 {demand:.3f} Gb/s　送达 {delivered:.3f} Gb/s　未送达 {max(0,demand-delivered):.3f} Gb/s　送达率 {100*delivered/max(demand,1e-12):.1f}%\n调度结果：{planning.get('action','等待调度')}　主要原因：{reason}\n时延：加权平均 {metrics['average_delay_ms']:.3f} ms　P95 {metrics['p95_delay_ms']:.3f} ms　最大 {metrics['maximum_delay_ms']:.3f} ms　有效馈电链路 {len(metrics['actual'])}/{len(metrics['planned'])}")
        if rebuild_history:self.history.setRowCount(0)
        history=self.frame_history if rebuild_history else ([self.frame_history[-1]] if append_history and self.frame_history else [])
        for event,_names in history:
            ep=event.get("planning") or {};ea=ep.get("active_evaluation") or {};em=_station_metrics(event,gid)
            ed=float(ep.get("delivered_gbps",0)) if gid is None else em["scheduled"]
            current_demand=float((ep.get("decision_context") or {}).get("traffic_demand_gbps",0)) if gid is None else float(em["demand"])
            eu=max(0,current_demand-ed)
            er=float(ea.get("route_available_ratio",0)) if gid is None else em["route_ratio"]
            switch_value=ep.get("switch_count",0) if gid is None else sum(bool(x.get("is_substitute")) for x in em["actual"])
            service_ratio=ed/max(current_demand,1e-12)
            reason=_delivery_gap_reason(ea,current_demand,ed,er)
            PlanningResultsWidget._fill(self.history,self.history.rowCount(),[f"{event['time_s']:.2f}s",f"{current_demand:.3f}",f"{ed:.3f}",f"{eu:.3f}",f"{100*service_ratio:.1f}%",f"{100*er:.1f}%",f"{em['p95_delay_ms']:.3f}",len(em["actual"]),switch_value,ep.get("action","--"),reason])
        while self.history.rowCount()>500:self.history.removeRow(0)
        self.deployment.setRowCount(0)
        rows=deployment.get("stations",[]);rows=rows if gid is None else [x for x in rows if int(x.get("gateway",-1))==gid]
        for i,row in enumerate(rows):
            PlanningResultsWidget._fill(self.deployment,i,[row["name"],row["visible_satellites"],f"{row['used_channels']}/{row['planned_channels']}",row["planned_channels"],row["recommended_antennas"],"满足" if row["minimum_connection_satisfied"] else "无连接"])
        self.links.setRowCount(0)
        actual_by_pair={(int(x["gateway"]),int(x["satellite"])):x for x in metrics["actual"]}
        states_by_pair={(int(x["gateway"]),int(x["satellite"])):x for x in active.get("link_states",[])}
        for i,planned_link in enumerate(metrics["planned"]):
            pair=(int(planned_link["gateway"]),int(planned_link["satellite"]))
            x=actual_by_pair.get(pair) or {};state=states_by_pair.get(pair) or {}
            gateway,satellite=pair;name=station_names[gateway] if gateway<len(station_names) else f"GW-{gateway+1:02d}"
            rate=float(x.get("rate_gbps",0));used=float(x.get("scheduled_gbps",0));elevation=state.get("elevation_deg")
            PlanningResultsWidget._fill(self.links,i,[name,f"SAT-{satellite+1:05d}",f"{float(elevation):.2f}°" if elevation is not None else "—",
                f"{float(x.get('rain_actual_db',state.get('rain_attenuation_db',0))):.2f}" if elevation is not None else "—",
                f"{float(x.get('snr_db',0)):.2f}" if rate>0 else "—",f"{rate:.3f}",f"{used:.3f}",f"{max(0.,rate-used):.3f}",state.get("status","状态未知")])
    def clear(self):
        for table in (self.history,self.deployment,self.links,self.routes):table.setRowCount(0)
        self.summary.setText("等待调度执行结果")
        self.latest_data=None;self.frame_history=[]


class PlanningEvaluationWidget(QWidget):
    """Explain whether planning met demand and which input condition constrained it."""
    def __init__(self):
        super().__init__();layout=QVBoxLayout(self);layout.setContentsMargins(12,48,12,12);layout.setSpacing(8)
        self.gateway_scope=GatewayScopeTabs();self.gateway_scope.currentChanged.connect(lambda _index:self._render(True,False))
        self.latest_data=None;self.station_names=[];self.frame_history=[]
        self.summary=QLabel("等待效果评价");self.summary.setObjectName("planningKpi");self.summary.setWordWrap(True);layout.addWidget(self.summary)
        layout.addWidget(_gateway_scope_row(self.gateway_scope))
        self.result_splitter=QSplitter(Qt.Orientation.Vertical);self.result_splitter.setObjectName("planningResultSplitter")
        chart_page=QWidget();chart_layout=QVBoxLayout(chart_page);chart_layout.setContentsMargins(0,0,0,0)
        self.business_page=QGroupBox("业务送达与容量（Gb/s）");business_layout=QVBoxLayout(self.business_page);self.chart=ResultTrendChart("business");business_layout.addWidget(self.chart,1)
        self.delay_page=QGroupBox("传播时延（ms）");delay_layout=QVBoxLayout(self.delay_page);self.delay_chart=ResultTrendChart("delay");delay_layout.addWidget(self.delay_chart,1)
        chart_layout.addWidget(self.business_page,1);chart_layout.addWidget(self.delay_page,1)
        event_page=QGroupBox("关键变化与影响");event_layout=QVBoxLayout(event_page);event_layout.setContentsMargins(8,8,8,8)
        self.events=QTableWidget(0,6);self.events.setHorizontalHeaderLabels(["时刻","影响来源","输入状态","规划/调度响应","结果影响","结论"]);PlanningResultsWidget._style_table(self.events);event_layout.addWidget(self.events,1)
        self.result_splitter.addWidget(chart_page);self.result_splitter.addWidget(event_page);self.result_splitter.setStretchFactor(0,3);self.result_splitter.setStretchFactor(1,2);self.result_splitter.setSizes([520,280]);layout.addWidget(self.result_splitter,1)
    def save_charts_image(self,path):
        self.chart.hover_index=None;self.delay_chart.hover_index=None
        images=[self.business_page.grab(),self.delay_page.grab()]
        combined=QPixmap(max(image.width() for image in images),sum(image.height() for image in images)+8)
        combined.fill(QColor("#071825"));painter=QPainter(combined)
        painter.drawPixmap(0,0,images[0]);painter.drawPixmap(0,images[0].height()+8,images[1]);painter.end()
        return combined.save(str(path),"PNG")
    def update_frame(self,data,_station_names):
        self.latest_data=data;self.station_names=list(_station_names);_sync_gateway_scope(self.gateway_scope,self.station_names)
        added=not self.frame_history or self.frame_history[-1].get("index")!=data.get("index")
        if added:
            self.frame_history.append(data);self.frame_history=self.frame_history[-600:]
        self._render(False,added)
    def _render(self,rebuild_history=False,append_history=False):
        if not self.latest_data:return
        data=self.latest_data;gid=self.gateway_scope.currentData();scope="全网" if gid is None else _gateway_name(self.station_names,gid)
        planning=data.get("planning") or {};active=planning.get("active_evaluation") or {};plan=planning.get("active_plan") or {};context=planning.get("decision_context") or {}
        metrics=_station_metrics(data,gid)
        demand=float(context.get("traffic_demand_gbps",plan.get("total_demand_gbps",0))) if gid is None else metrics["demand"]
        delivered=float(planning.get("delivered_gbps",0)) if gid is None else metrics["scheduled"]
        service=delivered/max(demand,1e-12) if demand else (1.0 if metrics["actual"] else 0.0)
        cap=float(active.get("capacity_achievement_ratio",0)) if gid is None else metrics["capacity_ratio"]
        route=float(active.get("route_available_ratio",0)) if gid is None else metrics["route_ratio"]
        unserved=max(0,demand-delivered)
        if gid is not None and not metrics["planned"]:source="该站本轮未分配规划链路"
        elif gid is not None and not metrics["actual"]:source="该站规划链路当前不可用"
        elif gid is not None and metrics["rain_max"]>=3:source="该站雨衰与星地链路容量"
        else:source=context.get("constraint_source") or "无显著约束"
        if gid is not None:
            state=f"规划链路 {len(metrics['planned'])} 条，实际链路 {len(metrics['actual'])} 条，最大雨衰 {metrics['rain_max']:.2f} dB"
            impact=f"经该站承载 {delivered:.3f} Gb/s，未承载 {unserved:.3f} Gb/s"
        elif route<.999:state=f"ISL {context.get('isl_count',0)} 条，可用路径 {route*100:.1f}%";impact="部分业务路径中断"
        elif cap<.999:state=f"GSL候选 {context.get('gsl_candidate_count',0)} 条，容量达成 {cap*100:.1f}%";impact="可用容量低于计划"
        elif unserved>.001:state=f"{context.get('traffic_satellite_count',0)} 颗卫星汇聚 {demand:.3f} Gb/s";impact=f"未服务 {unserved:.3f} Gb/s"
        else:state=f"ISL {context.get('isl_count',0)} 条，GSL候选 {context.get('gsl_candidate_count',0)} 条";impact="无显著约束"
        conclusion="满足" if service>=.99 else ("基本满足" if service>=.9 else "需优化")
        gap_reason=_delivery_gap_reason(active,demand,delivered,route)
        self.summary.setText(f"{scope} · 需求 {demand:.3f} Gb/s　送达 {delivered:.3f} Gb/s　未送达 {unserved:.3f} Gb/s　送达率 {service*100:.1f}%\n主要原因：{gap_reason}\n时延：加权平均 {metrics['average_delay_ms']:.3f} ms　P95 {metrics['p95_delay_ms']:.3f} ms　最大 {metrics['maximum_delay_ms']:.3f} ms")
        if gid is None and float(context.get("priority_demand_gbps",0))>0:
            priority_demand=float(context["priority_demand_gbps"])
            priority_delivered=min(priority_demand,float(context.get("priority_delivered_gbps",0)))
            self.summary.setText(self.summary.text()+f"\n重点区域业务：需求 {priority_demand:.3f} Gb/s　送达 {priority_delivered:.3f} Gb/s　送达率 {100*priority_delivered/priority_demand:.1f}%　平均路径时延 {float(context.get('priority_average_delay_ms',0)):.3f} ms")
        if rebuild_history:self.chart.clear();self.delay_chart.clear();self.events.setRowCount(0)
        history=self.frame_history if rebuild_history else ([self.frame_history[-1]] if append_history and self.frame_history else [])
        for event in history:
            ep=event.get("planning") or {};em=_station_metrics(event,gid);event_active=ep.get("active_evaluation") or {};event_context=ep.get("decision_context") or {}
            event_demand=float(event_context.get("traffic_demand_gbps",0)) if gid is None else float(em["demand"]);event_delivered=float(ep.get("delivered_gbps",0)) if gid is None else float(em["scheduled"])
            self.chart.append_values(event["time_s"],event_demand,event_delivered,em["capacity"]);self.delay_chart.append_values(event["time_s"],em["average_delay_ms"],em["p95_delay_ms"])
            event_service=event_delivered/max(event_demand,1e-12) if event_demand else 0.;event_route=float(event_active.get("route_available_ratio",0)) if gid is None else float(em["route_ratio"])
            if ep.get("planning_event") or bool(em["actual"])!=bool(em["planned"]) or em["rain_max"]>=3 or event_service<.99 or event_route<.999 or int(ep.get("connection_change_count",0))>0:
                esource=(ep.get("decision_context") or {}).get("constraint_source","无显著约束") if gid is None else ("雨衰影响" if em["rain_max"]>=3 else ("规划链路失效" if em["planned"] and not em["actual"] else "规划/调度更新"))
                einput=f"规划 {len(em['planned'])} 条，实际 {len(em['actual'])} 条，雨衰最大 {em['rain_max']:.2f} dB"
                event_reason=_delivery_gap_reason(event_active,event_demand,event_delivered,event_route)
                eimpact=f"需求 {event_demand:.3f}，送达 {event_delivered:.3f}，缺口 {max(0,event_demand-event_delivered):.3f} Gb/s；{event_reason}；P95 {em['p95_delay_ms']:.3f} ms"
                PlanningResultsWidget._fill(self.events,self.events.rowCount(),[f"{event['time_s']:.2f}s",esource,einput,ep.get("action","--"),eimpact,"正常" if event_service>=.99 and event_route>=.99 else "需关注"])
        while self.events.rowCount()>300:self.events.removeRow(0)
    def clear(self):
        self.chart.clear();self.delay_chart.clear();self.events.setRowCount(0);self.summary.setText("等待效果评价")
        self.latest_data=None;self.frame_history=[]


PRESET_STATIONS = {
    "北京": (39.9075, 116.39723), "上海": (31.22222, 121.45806),
    "广州": (23.125457, 113.257374), "成都": (30.66667, 104.06667),
    "西安": (34.289413, 108.940242), "乌鲁木齐": (43.8256, 87.6168),
    "三亚": (18.2528, 109.5119), "拉萨": (29.652, 91.1721),
    "佳木斯": (46.8, 130.3), "喀什": (39.5, 76.0),
}

# Coarse, dependency-free continental outlines used only by the built-in
# demonstration grid. Imported WorldPop/H3 data remains the accurate option.
DEMO_LAND_POLYGONS = (
    ((-168, 72), (-150, 60), (-135, 55), (-125, 48), (-117, 32), (-105, 20),
     (-90, 18), (-82, 25), (-75, 40), (-60, 48), (-55, 60), (-80, 72), (-120, 75)),
    ((-81, 12), (-70, 10), (-50, 5), (-35, -5), (-40, -25), (-55, -55),
     (-72, -50), (-80, -20)),
    ((-18, 36), (5, 37), (32, 31), (50, 12), (42, -12), (32, -35),
     (15, -35), (0, -25), (-12, 5)),
    ((-10, 36), (10, 55), (35, 70), (70, 75), (120, 72), (170, 62),
     (165, 48), (145, 42), (130, 30), (120, 20), (105, 5), (95, 8),
     (80, 22), (65, 25), (45, 38), (28, 42), (12, 42)),
    ((112, -10), (154, -10), (154, -40), (135, -44), (115, -35)),
    ((-73, 60), (-15, 60), (-20, 83), (-55, 84)),
)

# A lightweight mainland/nearby-island outline used to decide whether a
# traffic cell receives the China or outside-China demand coefficient.  It is
# intentionally independent of optional GIS packages so saved scenarios can
# be replayed on every installation.
CHINA_TRAFFIC_POLYGON = (
    (73.5, 39.5), (79.0, 49.0), (87.0, 49.2), (96.0, 42.5),
    (107.0, 42.5), (119.5, 53.6), (134.8, 48.3), (131.0, 42.0),
    (124.0, 39.0), (122.0, 30.0), (121.0, 22.0), (114.0, 21.0),
    (108.0, 18.0), (101.0, 21.0), (97.0, 28.0), (91.0, 27.0),
    (85.0, 29.0), (79.0, 32.0), (73.5, 39.5),
)


def _point_in_polygon(lon: float, lat: float, polygon) -> bool:
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = previous
        x2, y2 = current
        if ((y1 > lat) != (y2 > lat)) and (
                lon < (x2 - x1) * (lat - y1) / (y2 - y1) + x1):
            inside = not inside
        previous = current
    return inside


def is_demo_land(lat: float, lon: float) -> bool:
    return any(_point_in_polygon(lon, lat, polygon)
               for polygon in DEMO_LAND_POLYGONS)


def is_china_traffic_cell(lat: float, lon: float) -> bool:
    return _point_in_polygon(lon, lat, CHINA_TRAFFIC_POLYGON)


def geodetic_to_ecef(lat_deg, lon_deg, alt_m):
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    e2 = 2 * EARTH_FLATTENING - EARTH_FLATTENING**2
    sin_lat = math.sin(lat)
    n = EARTH_RADIUS_M / math.sqrt(1 - e2 * sin_lat * sin_lat)
    return ((n + alt_m) * math.cos(lat) * math.cos(lon),
            (n + alt_m) * math.cos(lat) * math.sin(lon),
            (n * (1 - e2) + alt_m) * sin_lat)


def parse_tle_epoch(line1: str) -> datetime:
    year = int(line1[18:20]); year += 2000 if year < 57 else 1900
    day = float(line1[20:32])
    return datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=day - 1)


def greenwich_sidereal_angle(utc_time: datetime) -> float:
    """Greenwich mean sidereal angle in radians for ECI-to-ECEF rotation."""
    if utc_time.tzinfo is None:
        utc_time=utc_time.replace(tzinfo=timezone.utc)
    timestamp=utc_time.astimezone(timezone.utc).timestamp()
    julian_date=timestamp/86400.0+2440587.5
    centuries=(julian_date-2451545.0)/36525.0
    degrees=(280.46061837+360.98564736629*(julian_date-2451545.0)
             +0.000387933*centuries*centuries-centuries**3/38710000.0)
    return math.radians(degrees%360.0)


def read_tle_records(path: Path):
    lines = [line.rstrip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    header_planes = header_per_plane = None
    if lines and len(lines[0].split()) == 2 and all(x.isdigit() for x in lines[0].split()):
        header_planes, header_per_plane = map(int, lines.pop(0).split())
    blocks, i = [], 0
    while i < len(lines):
        if lines[i].startswith("1 ") and i + 1 < len(lines) and lines[i + 1].startswith("2 "):
            name, line1, line2 = f"SAT-{len(blocks)}", lines[i], lines[i + 1]; i += 2
        elif i + 2 < len(lines) and lines[i + 1].startswith("1 ") and lines[i + 2].startswith("2 "):
            name, line1, line2 = lines[i], lines[i + 1], lines[i + 2]; i += 3
        else:
            raise ValueError(f"无法识别TLE第 {i + 1} 行附近的格式")
        blocks.append((name, line1, line2))
    if not blocks:
        raise ValueError("TLE文件中没有卫星")
    return blocks, header_planes, header_per_plane


def read_tle_file(path: Path):
    blocks, header_planes, header_per_plane = read_tle_records(path)
    epoch = parse_tle_epoch(blocks[0][1])
    satellites = [ephem.readtle(name, l1, l2) for name, l1, l2 in blocks]
    return satellites, epoch, header_planes, header_per_plane


def _tle_checksum(line):
    return str(sum(int(char) if char.isdigit() else (1 if char=="-" else 0) for char in line)%10)


def _tle_with_number(record,number):
    name,line1,line2=record
    line1=line1[:2]+f"{number:05d}"+line1[7:68]
    line2=line2[:2]+f"{number:05d}"+line2[7:68]
    return name,line1+_tle_checksum(line1),line2+_tle_checksum(line2)


def _analytic_tle_record(name,number,inclination_deg,raan_deg,phase_deg,mean_motion):
    line1=f"1 {number:05d}U 00000ABC 00001.00000000  .00000000  00000-0  00000+0 0    0"
    line2=f"2 {number:05d} {inclination_deg:8.4f} {raan_deg%360:8.4f} 0000001 {0.:8.4f} {phase_deg%360:8.4f} {mean_motion:11.8f}    0"
    return name,line1+_tle_checksum(line1),line2+_tle_checksum(line2)


def make_candidates(planes: int, per_plane: int, isl_pattern: str,
                    seam_enabled: bool, seam_after_plane: int):
    edges, seam_edges = [], []
    for plane in range(planes):
        start = plane * per_plane
        for j in range(per_plane):
            edges.append((start + j, start + (j + 1) % per_plane, 0))
    for plane in range(planes):
        next_plane = (plane + 1) % planes
        for j in range(per_plane):
            if isl_pattern == "3isl_brick" and (plane + j) % 2 != 0:
                continue
            edge = (plane * per_plane + j, next_plane * per_plane + j, 1)
            if seam_enabled and plane == seam_after_plane:
                seam_edges.append(edge)
            else:
                edges.append(edge)
    return edges, seam_edges


def line_clear_of_earth(a, b):
    ax, ay, az = a; dx, dy, dz = b[0] - ax, b[1] - ay, b[2] - az
    dd = dx * dx + dy * dy + dz * dz
    if dd == 0:
        return False
    q = max(0.0, min(1.0, -(ax * dx + ay * dy + az * dz) / dd))
    x, y, z = ax + q * dx, ay + q * dy, az + q * dz
    return x * x + y * y + z * z > EARTH_RADIUS_M**2

def compute_gsl_rows(xyz,stations,min_elevation_deg,max_gsl_km):
    started=time.perf_counter()
    import numpy as np
    rows=[];coordinates=np.asarray(xyz);max_distance=max_gsl_km*1000
    for gid,gs in enumerate(stations):
        lat,lon=math.radians(gs["lat"]),math.radians(gs["lon"]);ground=np.asarray(geodetic_to_ecef(gs["lat"],gs["lon"],gs.get("alt",0.0)));up=np.asarray((math.cos(lat)*math.cos(lon),math.cos(lat)*math.sin(lon),math.sin(lat)));offsets=coordinates-ground;distances=np.linalg.norm(offsets,axis=1);elevations=np.degrees(np.arcsin(np.clip((offsets@up)/np.maximum(distances,1.0),-1.0,1.0)));visible=elevations>=min_elevation_deg
        if max_distance>0:visible&=distances<=max_distance
        rows.extend([gid,int(sid),round(float(elevations[sid]),2)] for sid in np.flatnonzero(visible))
    return rows,round((time.perf_counter()-started)*1000,1)

def compute_traffic_rows(traffic_model,traffic_cells,positions,min_elevation_deg,frame_time):
    started=time.perf_counter()
    if traffic_model is None:return [],[],0.0
    visibility=calculate_visibility(traffic_cells,positions,min_elevation_deg);result=traffic_model.run_frame(frame_time,visibility);cell_by_id={cell.cell_id:cell for cell in traffic_cells}
    rows=[[item.cell_id,cell_by_id[item.cell_id].latitude_deg,cell_by_id[item.cell_id].longitude_deg,item.demand_weight,item.dense,item.unassigned_weight,[[sid,value] for sid,value in item.allocations.items()],item.local_hour,[[candidate.satellite_id,candidate.elevation_deg] for candidate in visibility.get(item.cell_id,())],item.dense_cluster_id] for item in result.cells]
    satellite_rows=[[sid,load.assigned_weight,load.served_cell_count] for sid,load in result.satellites.items() if load.assigned_weight>0.0]
    return rows,satellite_rows,round((time.perf_counter()-started)*1000,1)

def priority_traffic_by_satellite(traffic_rows,regions):
    """Identify priority traffic by its ground-cell origin, not satellite position."""
    if not regions:return {}
    from constellation_sim.algorithm.planning import _surface_distance_km
    loads={}
    for row in traffic_rows:
        if not any(_surface_distance_km(float(row[1]),float(row[2]),float(region['lat']),float(region['lon']))<=float(region['radius_km']) for region in regions):
            continue
        for satellite,amount in row[6]:
            satellite=int(satellite)
            loads[satellite]=loads.get(satellite,0.)+float(amount)
    return loads

def read_traffic_cache_frame(source,frame_index):
    started=time.perf_counter()
    if isinstance(source,(list,tuple)):
        blob=source[frame_index] if 0<=frame_index<len(source) else None
    else:
        row=source.execute("SELECT blob FROM frames WHERE frame_index=?",(frame_index,)).fetchone();blob=row[0] if row else None
    if blob is None:raise RuntimeError(f"业务时变缓存缺少第{frame_index}帧")
    if isinstance(blob,tuple) and len(blob)==2:traffic,satellite_traffic=blob
    else:traffic,satellite_traffic=marshal.loads(zlib.decompress(blob))
    return traffic,satellite_traffic,round((time.perf_counter()-started)*1000,1)


class SimulationWorker(QObject):
    prepared = Signal(dict)
    frame_ready = Signal(dict)
    progress = Signal(int, str)
    failed = Signal(str)
    finished = Signal()
    paused_changed = Signal(bool)
    display_frame_ready = Signal(dict)
    offline_planning_ready = Signal(list)

    def __init__(self, config: SimulationConfig, result_store=None):
        super().__init__(); self.config = config; self._stop = threading.Event()
        self._control_lock = threading.Lock(); self._requested_frame = None
        self._paused = True; self._speed = config.speed;self.result_store=result_store;self._latest_full_frame=None
        self._display_options={"planes":config.planes,"count":config.display_plane_count,"center":config.center_plane,"page":0,"bounds":None,"station_count":len(config.stations)}
        self._planning_strategy_request=(.10,.90)
        self._planning_selection_request=None
        self._planning_reserve_request=None
        self._scheduling_request=None
        self._offline_plan_requested=False;self._offline_catalog=[];self._offline_grade="轻雨";self._installed_offline_key=None

    def set_display_options(self,options):
        with self._control_lock:self._display_options=dict(options)

    @Slot(float,float)
    def set_planning_weights(self,capacity_weight,delay_weight):
        total=max(0.,float(capacity_weight))+max(0.,float(delay_weight))
        if total<=0:return
        with self._control_lock:
            self._planning_strategy_request=(max(0.,float(capacity_weight))/total,max(0.,float(delay_weight))/total)

    def latest_full_frame(self):
        with self._control_lock:return self._latest_full_frame

    @Slot()
    def request_offline_planning(self,capacity_weight=None,delay_weight=None,capacity_reserve_ratio=None,selection_strategy=None):
        with self._control_lock:
            self._offline_plan_requested=True
            if capacity_weight is not None and delay_weight is not None:self._planning_strategy_request=(float(capacity_weight),float(delay_weight))
            if capacity_reserve_ratio is not None:self._planning_reserve_request=max(0.,float(capacity_reserve_ratio))
            if selection_strategy is not None:self._planning_selection_request=str(selection_strategy)

    def set_scheduling_parameters(self,evaluation_s,adjust_s,route_s,hold_s,capacity_threshold,switching_loss):
        with self._control_lock:self._scheduling_request=(float(evaluation_s),float(adjust_s),float(route_s),float(hold_s),float(capacity_threshold),float(switching_loss))

    @Slot(str)
    def set_offline_rain_grade(self,grade):
        with self._control_lock:self._offline_grade=str(grade);self._installed_offline_key=None

    @Slot()
    def run(self):
        try:
            cfg = self.config
            analytic_walker = cfg.source == "walker" and cfg.eccentricity <= 0.000001
            analytic_params = [];tle_records=[]
            self.progress.emit(2, "准备阶段 1/4：检查场景参数")
            if cfg.source == "walker":
                if cfg.walker_mode not in ("multi_shell","document_4_shell") and cfg.satellite_count%cfg.planes:
                    raise ValueError("单壳层Walker卫星总数必须能被轨道面数量整除")
                shells=cfg.walker_shells if cfg.walker_mode in ("multi_shell","document_4_shell") and cfg.walker_shells else [{"name":"L1","altitude_km":cfg.altitude_km,"inclination_deg":cfg.inclination_deg,"planes":cfg.planes,"per_plane":cfg.satellite_count//cfg.planes}]
                cfg.satellite_count=sum(int(x["planes"])*int(x["per_plane"]) for x in shells);cfg.planes=sum(int(x["planes"]) for x in shells)
                constellation_spec={"cache_version":1,"shells":shells,"layout":cfg.walker_layout,
                                    "eccentricity":cfg.eccentricity,"arg_perigee_deg":cfg.arg_perigee_deg}
                constellation_key=hashlib.sha256(json.dumps(constellation_spec,sort_keys=True,ensure_ascii=False,separators=(",",":")).encode("utf-8")).hexdigest()[:24]
                constellation_root=ROOT/"simulation_cache"/"constellation";constellation_root.mkdir(parents=True,exist_ok=True)
                constellation_path=constellation_root/f"{constellation_key}.tles"
                cached_tle_records=[];cached_satellites=[];cached_epoch=None;constellation_cache_reused=False
                if constellation_path.is_file():
                    try:
                        cached_tle_records=read_tle_records(constellation_path)[0]
                        if len(cached_tle_records)==cfg.satellite_count:
                            if not analytic_walker:cached_satellites,cached_epoch,_,_=read_tle_file(constellation_path)
                            constellation_cache_reused=analytic_walker or len(cached_satellites)==cfg.satellite_count
                    except (OSError,ValueError,TypeError):pass
                if not constellation_cache_reused:cached_tle_records=[];cached_satellites=[]
                per_plane=int(shells[0]["per_plane"]);satellites=[];satellite_planes=[];satellite_slots=[];satellite_shells=[];satellite_labels=[];shell_runs=[]
                method="向量化圆轨道参数" if analytic_walker else "Walker TLE"
                self.progress.emit(10, f"准备阶段 2/4：正在生成 {cfg.satellite_count:,} 颗卫星的{method}")
                with tempfile.TemporaryDirectory(prefix="hypatia_qt_") as tmp:
                    sat_offset=plane_offset=0;epoch=None;perigees=[];apogees=[]
                    for shell_index,shell in enumerate(shells):
                        shell_planes=int(shell["planes"]);shell_per_plane=int(shell["per_plane"]);altitude=float(shell["altitude_km"]);inclination=float(shell["inclination_deg"])
                        semi_major_m=EARTH_RADIUS_M+altitude*1000;perigee=(semi_major_m*(1-cfg.eccentricity)-EARTH_RADIUS_M)/1000;apogee=(semi_major_m*(1+cfg.eccentricity)-EARTH_RADIUS_M)/1000
                        if perigee<100:raise ValueError(f"壳层 {shell.get('name',shell_index+1)} 的近地点高度仅 {perigee:.1f} km，低于100 km")
                        angular_rate=math.sqrt(MU/(semi_major_m**3));mean_motion=angular_rate*86400/(2*math.pi)
                        if analytic_walker:
                            shell_sats=[None]*(shell_planes*shell_per_plane);shell_epoch=datetime(2000,1,1,tzinfo=timezone.utc)
                            raan_span=math.pi if cfg.walker_layout=="star" else 2*math.pi
                            for local_plane in range(shell_planes):
                                raan=raan_span*local_plane/shell_planes
                                for slot in range(shell_per_plane):
                                    phase=2*math.pi*(slot/shell_per_plane+local_plane/(shell_planes*shell_per_plane))
                                    analytic_params.append((semi_major_m,math.radians(inclination),raan,phase,angular_rate))
                                    label=f"{shell.get('name',f'L{shell_index+1}')}-P{local_plane+1:03d}-S{slot+1:03d}"
                                    if not constellation_cache_reused:tle_records.append(_analytic_tle_record(label,sat_offset+local_plane*shell_per_plane+slot+1,inclination,math.degrees(raan),math.degrees(phase),mean_motion))
                        elif constellation_cache_reused:
                            shell_sats=cached_satellites[sat_offset:sat_offset+shell_planes*shell_per_plane];shell_epoch=cached_epoch
                        else:
                            tle_path=Path(tmp)/f"walker_shell_{shell_index+1}.tles"
                            satgen.generate_tles_from_scratch_manual(str(tle_path),f"QT-{shell.get('name',shell_index+1)}",shell_planes,shell_per_plane,False,inclination,cfg.eccentricity,cfg.arg_perigee_deg,mean_motion,180.0 if cfg.walker_layout=="star" else 360.0)
                            shell_sats,shell_epoch,_,_=read_tle_file(tle_path)
                            tle_records.extend(_tle_with_number(record,sat_offset+index+1) for index,record in enumerate(read_tle_records(tle_path)[0]))
                        epoch=epoch or shell_epoch;satellites.extend(shell_sats)
                        for local_plane in range(shell_planes):
                            satellite_planes.extend([plane_offset+local_plane]*shell_per_plane);satellite_slots.extend(range(shell_per_plane));satellite_labels.extend(f"{shell.get('name',f'L{shell_index+1}')}-P{local_plane+1:03d}-S{slot+1:03d}" for slot in range(shell_per_plane))
                            satellite_shells.extend([shell_index]*shell_per_plane)
                        shell_runs.append({**shell,"satellite_offset":sat_offset,"plane_offset":plane_offset});sat_offset+=len(shell_sats);plane_offset+=shell_planes;perigees.append(perigee);apogees.append(apogee)
                    perigee_altitude_km=min(perigees);apogee_altitude_km=max(apogees)
                if constellation_cache_reused:tle_records=cached_tle_records
            else:
                self.progress.emit(10, "准备阶段 2/4：正在读取并解析TLE文件")
                satellites, epoch, file_planes, file_per_plane = read_tle_file(Path(cfg.tle_path))
                if cfg.satellite_count > 0:
                    satellites = satellites[:cfg.satellite_count]
                tle_records=read_tle_records(Path(cfg.tle_path))[0][:len(satellites)]
                if file_planes and len(satellites) == file_planes * file_per_plane:
                    cfg.planes = file_planes
                if len(satellites) % cfg.planes:
                    raise ValueError("导入的卫星数量必须能被轨道面数量整除")
                per_plane = len(satellites) // cfg.planes
                satellite_planes=[sid//per_plane for sid in range(len(satellites))];satellite_slots=[sid%per_plane for sid in range(len(satellites))]
                satellite_shells=[0]*len(satellites)
                satellite_labels=[f"P{satellite_planes[sid]+1:03d}-S{satellite_slots[sid]+1:03d}" for sid in range(len(satellites))]
                shell_runs=[{"name":"TLE","planes":cfg.planes,"per_plane":per_plane,"satellite_offset":0,"plane_offset":0}]
            if len(tle_records)!=len(satellites):raise ValueError("卫星 TLE 数量与轨道对象数量不一致")
            tle_asset=Path(cfg.scenario_directory)/"base_data"/"satellites.tles"
            tle_asset.parent.mkdir(parents=True,exist_ok=True)
            if cfg.source=="walker" and constellation_cache_reused:
                if tle_asset!=constellation_path:shutil.copy2(constellation_path,tle_asset)
            else:
                temporary_tle=tle_asset.with_suffix(".building.tles")
                tle_header=[f"{cfg.planes} {len(satellites)//cfg.planes}"] if len(satellites)%cfg.planes==0 and all(int(shell["per_plane"])==len(satellites)//cfg.planes for shell in shell_runs) else []
                temporary_tle.write_text("\n".join((*tle_header,*(line for record in tle_records for line in record)))+"\n",encoding="utf-8")
                temporary_tle.replace(tle_asset)
            if cfg.source=="walker" and not constellation_cache_reused:shutil.copy2(tle_asset,constellation_path)
            tle_asset_hash=hashlib.sha256(tle_asset.read_bytes()).hexdigest()
            shared_tle_root=ROOT/"simulation_cache"/"tle";shared_tle_root.mkdir(parents=True,exist_ok=True)
            shared_tle_asset=shared_tle_root/f"{tle_asset_hash[:24]}.tles"
            if not shared_tle_asset.is_file():shutil.copy2(tle_asset,shared_tle_asset)
            self.progress.emit(45, f"准备阶段 2/4完成：已建立 {len(satellites):,} 个轨道传播对象")
            traffic_cells = []
            traffic_model = None
            if cfg.traffic_enabled:
                if cfg.traffic_grid_path:
                    traffic_cells = load_cells_csv(cfg.traffic_grid_path)
                else:
                    # Built-in global demonstration grid. Import a WorldPop/H3
                    # CSV when real population weights are required.
                    traffic_cells = []
                    for lat_index, lat in enumerate(range(-85, 86, 5)):
                        area_weight = max(0.04, math.cos(math.radians(lat)) ** 2)
                        for lon_index, lon in enumerate(range(-175, 180, 5)):
                            if not is_demo_land(float(lat), float(lon)):
                                continue
                            variation = 0.72 + 0.28 * (
                                0.5 + 0.5 * math.sin(math.radians(lon * 3 + lat * 2)))
                            weight = area_weight * variation
                            cell_id = f"G{lat_index:02d}-{lon_index:02d}"
                            traffic_cells.append(TrafficCell(
                                cell_id, float(lat), float(lon), weight, weight))
                outside_scale = max(0.0, cfg.traffic_outside_china_percent) / 100.0
                scoped_cells = []
                for cell in traffic_cells:
                    scale = 1.0 if is_china_traffic_cell(
                        cell.latitude_deg, cell.longitude_deg) else outside_scale
                    if scale > 0.0:
                        scoped_cells.append(TrafficCell(
                            cell.cell_id, cell.latitude_deg, cell.longitude_deg,
                            cell.base_weight * scale, cell.population * scale,
                            cell.enabled))
                traffic_cells = scoped_cells
                traffic_model = TrafficModel(traffic_cells, TrafficModelConfig(
                    dense_percentile=cfg.traffic_dense_percentile,
                    dense_max_satellites=cfg.traffic_dense_max_satellites,
                    dense_cluster_radius_km=cfg.traffic_cluster_radius_km,
                    min_elevation_deg=cfg.min_elevation_deg,
                    random_sigma=cfg.traffic_random_sigma,
                    total_reference_gbps=cfg.traffic_total_gbps,
                ))
            if cfg.isl_pattern == "3isl_brick" and any(int(x["planes"])%2 or int(x["per_plane"])%2 for x in shell_runs):
                raise ValueError("3-ISL固定砖墙构型要求每个壳层的轨道面数和每轨卫星数均为偶数")
            if cfg.seam_enabled and not 0 <= cfg.seam_after_plane < cfg.planes:
                raise ValueError(f"缝隙起始轨道面必须在1到{cfg.planes}之间")
            self.progress.emit(55, f"准备阶段 3/4：正在生成{('3-ISL固定砖墙' if cfg.isl_pattern == '3isl_brick' else '4-ISL规则网格')}候选")
            import numpy as np
            cache_root=ROOT/"simulation_cache";cache_root.mkdir(parents=True,exist_ok=True)
            topology_spec={"cache_version":1,"shells":[[int(shell["planes"]),int(shell["per_plane"]),int(shell["satellite_offset"])] for shell in shell_runs],
                           "isl_pattern":cfg.isl_pattern,"seam_enabled":cfg.seam_enabled}
            topology_key=hashlib.sha256(json.dumps(topology_spec,sort_keys=True,separators=(",",":")).encode("utf-8")).hexdigest()[:24]
            topology_root=cache_root/"topology";topology_root.mkdir(parents=True,exist_ok=True)
            topology_path=topology_root/f"{topology_key}.npz";topology_cache_reused=False
            if topology_path.is_file():
                try:
                    with np.load(topology_path) as stored_topology:
                        candidates=stored_topology["candidates"].tolist();seam_candidates=stored_topology["seams"].tolist()
                    topology_cache_reused=True
                except (OSError,ValueError,KeyError):topology_cache_reused=False
            if not topology_cache_reused:
                candidates=[];seam_candidates=[]
                for shell in shell_runs:
                    local,local_seams=make_candidates(int(shell["planes"]),int(shell["per_plane"]),cfg.isl_pattern,cfg.seam_enabled,int(shell["planes"])-1)
                    offset=int(shell["satellite_offset"]);candidates.extend((a+offset,b+offset,k) for a,b,k in local);seam_candidates.extend((a+offset,b+offset,k) for a,b,k in local_seams)
                temporary_topology=topology_path.with_suffix(".building.npz")
                np.savez_compressed(temporary_topology,candidates=np.asarray(candidates,dtype=np.int64).reshape(-1,3),seams=np.asarray(seam_candidates,dtype=np.int64).reshape(-1,3))
                temporary_topology.replace(topology_path)
            if any(satellite_shells[a]!=satellite_shells[b] for a,b,_kind in candidates+seam_candidates):
                raise RuntimeError("ISL候选生成错误：检测到跨壳层链路")
            expected_same=sum(int(shell["planes"])*int(shell["per_plane"]) for shell in shell_runs)
            actual_same=sum(kind==0 for _a,_b,kind in candidates)
            if actual_same!=expected_same:
                raise RuntimeError(f"同轨ISL候选不完整：应为 {expected_same} 条，实际 {actual_same} 条")
            diagnostic_path=str(Path(cfg.result_directory)/"time_variation.jsonl") if cfg.result_directory else ""
            append_diagnostic(diagnostic_path,"isl_candidates",shell_count=len(shell_runs),same_candidates=actual_same,cross_candidates=sum(kind==1 for _a,_b,kind in candidates),seam_candidates=len(seam_candidates),cross_shell_candidates=0)
            candidate_array=np.asarray(candidates,dtype=np.int64) if candidates else np.empty((0,3),dtype=np.int64)
            modcod_table = load_modcod_table(cfg.modcod_table_path) if cfg.planning_enabled else DEFAULT_MODCOD_TABLE
            planner = SlowFastLoopPlanner(cfg.stations, PlannerConfig(
                planning_period_s=cfg.planning_period_s,
                planning_horizon_s=cfg.planning_horizon_s,
                planning_slot_s=cfg.planning_slot_s,
                fast_evaluation_period_s=cfg.fast_evaluation_period_s,
                fast_adjust_period_s=cfg.fast_adjust_period_s,
                route_update_period_s=cfg.route_update_period_s,
                freeze_window_s=cfg.planning_freeze_window_s,
                bandwidth_mhz=cfg.feeder_bandwidth_mhz,
                snr_margin_db=cfg.snr_margin_db,
                feeder_frequency_ghz=cfg.feeder_frequency_ghz,
                feeder_rolloff=cfg.feeder_rolloff,
                satellite_eirp_dbw=cfg.satellite_eirp_dbw,
                gateway_gt_dbk=cfg.gateway_gt_dbk,
                gateway_array_loss_db=cfg.gateway_array_loss_db,
                reference_satellite_altitude_km=cfg.altitude_km,
                capacity_threshold_ratio=cfg.capacity_threshold_percent / 100.0,
                switching_loss_ratio=cfg.switching_loss_percent / 100.0,
                modcod_table=modcod_table,
                rain_enabled=cfg.rain_enabled,
                rain_sense_sigma=cfg.rain_sense_sigma,
                rain_forecast_hours=cfg.rain_forecast_hours,
                rain_use_p90=cfg.rain_use_p90,
                traffic_landing_ratio=cfg.traffic_landing_percent / 100.0,
                geo_interference_enabled=cfg.geo_interference_enabled,
                geo_satellite_count=cfg.geo_satellite_count,
                gso_i_over_n_db=cfg.gso_i_over_n_db,
                gso_protection_angle_deg=cfg.gso_protection_angle_deg,
                capacity_weight=cfg.planning_capacity_weight,
                delay_weight=cfg.planning_delay_weight,
                selection_strategy=cfg.planning_selection_strategy,
                capacity_reserve_ratio=cfg.planning_capacity_reserve_percent / 100.0,
                priority_task_regions=tuple(cfg.priority_task_regions),
                planning_exit_limit_per_source=8,
                realtime_exit_limit_per_source=8,
            )) if cfg.planning_enabled else None
            self.progress.emit(80, f"准备阶段 3/4完成：有效候选 {len(candidates):,} 条，缝隙禁用候选 {len(seam_candidates):,} 条")
            if not 0 <= cfg.center_plane < cfg.planes:
                raise ValueError(f"中心轨道面必须在 1 到 {cfg.planes} 之间")
            total_frames = max(1, int(math.floor(cfg.duration_s / cfg.step_s)) + 1)
            orbit_cache=None;orbit_cache_path="";orbit_cache_reused=False;cache_key="";orbit_cache_bytes=total_frames*len(satellites)*6*4
            cache_limit_bytes=1024*1024*1024
            # Keep an on-disk orbit cache even when it is too large to load
            # into RAM.  A memory map avoids recomputing the orbit and also
            # lets dependent traffic and gateway stages reuse their own cache.
            if total_frames and satellites:
                import numpy as np
                # Content-addressed stage assets live outside the full-scenario
                # directory, so unrelated parameter edits can reuse them.
                tle_signature=""
                if cfg.source=="tle" and Path(cfg.tle_path).is_file():
                    tle_signature=file_content_hash(cfg.tle_path)
                cache_spec={"cache_version":2,"source":cfg.source,"tle":tle_signature,"shells":cfg.walker_shells,"count":len(satellites),"planes":cfg.planes,"layout":cfg.walker_layout,"altitude":cfg.altitude_km,"inclination":cfg.inclination_deg,"eccentricity":cfg.eccentricity,"arg_perigee":cfg.arg_perigee_deg,"duration":cfg.duration_s,"step":cfg.step_s}
                cache_key=hashlib.sha256(json.dumps(cache_spec,sort_keys=True,ensure_ascii=False).encode("utf-8")).hexdigest()[:24]
                orbit_root=cache_root/"orbit";orbit_root.mkdir(parents=True,exist_ok=True)
                cache_file=orbit_root/f"{cache_key}.npy";meta_file=orbit_root/f"{cache_key}.json"
                if cache_file.is_file() and meta_file.is_file():
                    try:
                        meta=json.loads(meta_file.read_text(encoding="utf-8"))
                        if (meta.get("shape")==[total_frames,len(satellites),6]
                                and meta.get("dtype")=="float32"
                                and meta.get("spec")==cache_spec):
                            orbit_cache=np.load(cache_file,mmap_mode="r" if orbit_cache_bytes>cache_limit_bytes else None)
                            orbit_cache_path=str(cache_file);orbit_cache_reused=True
                    except Exception:orbit_cache=None
                if orbit_cache is None:
                    self.progress.emit(82,f"准备阶段 4/4：预计算 {total_frames:,} 帧轨道位置（约 {orbit_cache_bytes/1024/1024:.0f} MB）")
                    temporary=cache_file.with_suffix(".building.npy");orbit_cache=np.lib.format.open_memmap(temporary,mode="w+",dtype="float32",shape=(total_frames,len(satellites),6))
                    cache_observer=ephem.Observer();cache_observer.lat="0";cache_observer.lon="0";cache_observer.elevation=0
                    for cache_index in range(total_frames):
                        if self._stop.is_set():return
                        cache_time=min(cache_index*cfg.step_s,cfg.duration_s)
                        if analytic_walker:
                            params=np.asarray(analytic_params,dtype=float);radius=params[:,0];inc=params[:,1];raan=params[:,2];u=params[:,3]+params[:,4]*cache_time
                            cos_u=np.cos(u);sin_u=np.sin(u);cos_o=np.cos(raan);sin_o=np.sin(raan);cos_i=np.cos(inc);sin_i=np.sin(inc)
                            x=radius*(cos_o*cos_u-sin_o*sin_u*cos_i);y=radius*(sin_o*cos_u+cos_o*sin_u*cos_i);z=radius*sin_u*sin_i
                            earth_angle=greenwich_sidereal_angle(epoch)+7.2921150e-5*cache_time;ce=math.cos(earth_angle);se=math.sin(earth_angle);xe=ce*x+se*y;ye=-se*x+ce*y
                            orbit_cache[cache_index,:,0]=np.degrees(np.arctan2(z,np.sqrt(xe*xe+ye*ye)));orbit_cache[cache_index,:,1]=np.degrees(np.arctan2(ye,xe));orbit_cache[cache_index,:,2]=radius-EARTH_RADIUS_M;orbit_cache[cache_index,:,3]=xe;orbit_cache[cache_index,:,4]=ye;orbit_cache[cache_index,:,5]=z
                        else:
                            cache_observer.date=epoch+timedelta(seconds=cache_time)
                            for sid,satellite in enumerate(satellites):
                                satellite.compute(cache_observer);lat=math.degrees(float(satellite.sublat));lon=math.degrees(float(satellite.sublong));alt=float(satellite.elevation);x,y,z=geodetic_to_ecef(lat,lon,alt);orbit_cache[cache_index,sid]=lat,lon,alt,x,y,z
                        if cache_index%max(1,total_frames//50)==0:self.progress.emit(82+round(15*(cache_index+1)/total_frames),f"准备阶段 4/4：轨道缓存 {cache_index+1:,}/{total_frames:,} 帧")
                    orbit_cache.flush();del orbit_cache
                    if cache_file.exists():cache_file.unlink()
                    temporary.replace(cache_file);meta_file.write_text(json.dumps({"shape":[total_frames,len(satellites),6],"dtype":"float32","spec":cache_spec},ensure_ascii=False),encoding="utf-8");orbit_cache=np.load(cache_file,mmap_mode="r" if orbit_cache_bytes>cache_limit_bytes else None);orbit_cache_path=str(cache_file)
            isl_model_spec={"cache_version":1,"topology_key":topology_key,"orbit_key":cache_key,
                            "max_isl_km":cfg.max_isl_km,"latitude_limit_deg":cfg.latitude_limit_deg,
                            "seam_after_plane":cfg.seam_after_plane,"walker_layout":cfg.walker_layout}
            isl_model_key=hashlib.sha256(json.dumps(isl_model_spec,sort_keys=True,separators=(",",":")).encode("utf-8")).hexdigest()[:24]
            isl_model_root=cache_root/"isl_model";isl_model_root.mkdir(parents=True,exist_ok=True)
            isl_model_file=isl_model_root/f"{isl_model_key}.json";isl_model_reused=False
            if isl_model_file.is_file():
                try:isl_model_reused=json.loads(isl_model_file.read_text(encoding="utf-8")).get("spec")==isl_model_spec
                except (OSError,ValueError,TypeError):pass
            if not isl_model_reused:
                temporary_isl=isl_model_file.with_suffix(".building.json")
                temporary_isl.write_text(json.dumps({"spec":isl_model_spec},ensure_ascii=False,separators=(",",":")),encoding="utf-8")
                temporary_isl.replace(isl_model_file)
            traffic_cache_connection=None;traffic_cache_blobs=None;traffic_cache_path="";traffic_cache_reused=False;traffic_key="";traffic_source_signature="";compressed_bytes=0;traffic_update_period_s=30.0;traffic_keyframe_count=0
            if traffic_model is not None and orbit_cache is not None:
                traffic_update_period_s=30.0
                traffic_stride=max(1,int(math.ceil(traffic_update_period_s/cfg.step_s)))
                traffic_keyframes=list(range(0,total_frames,traffic_stride))
                if traffic_keyframes[-1]!=total_frames-1:traffic_keyframes.append(total_frames-1)
                traffic_keyframe_count=len(traffic_keyframes)
                traffic_source_signature=""
                if cfg.traffic_grid_path and Path(cfg.traffic_grid_path).is_file():
                    traffic_source_signature=file_content_hash(cfg.traffic_grid_path)
                traffic_spec={"cache_version":3,"access_split_policy":"normal2_dense4","orbit_key":cache_key,"grid":traffic_source_signature,"cells":len(traffic_cells),"outside_china_percent":cfg.traffic_outside_china_percent,"min_elevation":cfg.min_elevation_deg,"dense_percentile":cfg.traffic_dense_percentile,"dense_max_satellites":cfg.traffic_dense_max_satellites,"random_sigma":cfg.traffic_random_sigma,"total_gbps":cfg.traffic_total_gbps,"cluster_radius":cfg.traffic_cluster_radius_km,"frames":total_frames,"step":cfg.step_s,"update_period_s":traffic_update_period_s,"keyframes":traffic_keyframes}
                traffic_key=hashlib.sha256(json.dumps(traffic_spec,sort_keys=True,ensure_ascii=False).encode("utf-8")).hexdigest()[:24]
                traffic_root=cache_root/"traffic";traffic_root.mkdir(parents=True,exist_ok=True)
                traffic_file=traffic_root/f"{traffic_key}.sqlite";traffic_cache_path=str(traffic_file)
                traffic_cache_connection=sqlite3.connect(str(traffic_file),check_same_thread=False);traffic_cache_connection.execute("PRAGMA journal_mode=WAL");traffic_cache_connection.execute("PRAGMA synchronous=NORMAL");traffic_cache_connection.execute("CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)");traffic_cache_connection.execute("CREATE TABLE IF NOT EXISTS frames(frame_index INTEGER PRIMARY KEY,blob BLOB NOT NULL)")
                encoded_spec=json.dumps(traffic_spec,sort_keys=True,ensure_ascii=False,separators=(",",":"))
                stored=traffic_cache_connection.execute("SELECT value FROM metadata WHERE key='spec'").fetchone();count=traffic_cache_connection.execute("SELECT COUNT(*) FROM frames").fetchone()[0]
                traffic_cache_reused=bool(stored and stored[0]==encoded_spec and count==traffic_keyframe_count)
                if not traffic_cache_reused:
                    if stored and stored[0]!=encoded_spec:
                        traffic_cache_connection.execute("DELETE FROM frames");count=0
                    traffic_cache_connection.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES('spec',?)",(encoded_spec,));traffic_cache_connection.commit()
                    existing={row[0] for row in traffic_cache_connection.execute("SELECT frame_index FROM frames")};missing=[index for index in traffic_keyframes if index not in existing]
                    completed=traffic_keyframe_count-len(missing)
                    # Business preparation runs sequentially in this worker.
                    # The cache remains sparse (one keyframe every 30 seconds),
                    # avoiding process startup, IPC copies and scheduling stalls.
                    self.progress.emit(82+round(15*completed/traffic_keyframe_count),f"准备阶段 4/4：顺序计算业务关键帧 {completed:,}/{traffic_keyframe_count:,}…")
                    for cache_index in missing:
                        if self._stop.is_set():return
                        cached=np.asarray(orbit_cache[cache_index]);position_array=np.empty((len(satellites),5),dtype=np.float64);position_array[:,0]=np.arange(len(satellites));position_array[:,1]=satellite_planes;position_array[:,2:5]=cached[:,0:3]
                        traffic_rows,satellite_rows,_=compute_traffic_rows(traffic_model,traffic_cells,position_array,cfg.min_elevation_deg,epoch+timedelta(seconds=min(cache_index*cfg.step_s,cfg.duration_s)))
                        blob=zlib.compress(marshal.dumps((traffic_rows,satellite_rows)),1);traffic_cache_connection.execute("INSERT OR REPLACE INTO frames(frame_index,blob) VALUES(?,?)",(cache_index,blob));completed+=1
                        if completed%5==0:traffic_cache_connection.commit()
                        if completed%max(1,traffic_keyframe_count//50)==0:self.progress.emit(82+round(15*completed/traffic_keyframe_count),f"准备阶段 4/4：业务关键帧 {completed:,}/{traffic_keyframe_count:,}（顺序计算）")
                    traffic_cache_connection.commit()
                # SQLite random reads and page faults caused periodic 0.5--0.6 s
                # traffic-stage stalls during playback.  The compressed cache is
                # immutable after preparation, so preload it sequentially when its
                # size is reasonable.  Frames remain compressed and are decoded on
                # demand, keeping memory bounded while removing disk latency.
                compressed_bytes=traffic_cache_connection.execute("SELECT COALESCE(SUM(LENGTH(blob)),0) FROM frames").fetchone()[0]
                if compressed_bytes<=512*1024*1024:
                    self.progress.emit(97,f"准备阶段 4/4：正在解码 {traffic_keyframe_count} 个业务关键帧（{compressed_bytes/1024/1024:.1f} MB）")
                    decoded_keyframes={}
                    for cache_index,blob in traffic_cache_connection.execute("SELECT frame_index,blob FROM frames ORDER BY frame_index"):
                        if cache_index in traffic_keyframes:decoded_keyframes[cache_index]=marshal.loads(zlib.decompress(blob))
                    if len(decoded_keyframes)==traffic_keyframe_count:
                        traffic_cache_blobs=[None]*total_frames;active_key=traffic_keyframes[0];next_key_index=1
                        for frame_index in range(total_frames):
                            if next_key_index<len(traffic_keyframes) and frame_index>=traffic_keyframes[next_key_index]:active_key=traffic_keyframes[next_key_index];next_key_index+=1
                            traffic_cache_blobs[frame_index]=decoded_keyframes[active_key]
            gateway_deployment={"status":"使用手动设置资源","algorithm":"手动设置","mode":"specified","station_count":len(cfg.stations),"planned_stations":cfg.stations,"stations":[]}
            gateway_cache_reused=False;gateway_key="";deployment_path=None
            if cfg.gateway_planning_enabled or cfg.gateway_planning_mode=="automatic":
                representative_rates=sorted(float(row.rate_gbps) if row.rate_gbps is not None else cfg.feeder_bandwidth_mhz*float(row.spectral_efficiency)/1000.0 for row in modcod_table)
                representative_channel_rate=representative_rates[len(representative_rates)//2] if representative_rates else cfg.feeder_bandwidth_mhz/1000.0
                station_channel_limit=cfg.gateway_max_antennas if cfg.gateway_planning_mode=="automatic" else max((int(site.get("channel_count",site.get("antenna_count",1))) for site in cfg.stations),default=1)
                effective_station_capacity=max(0.001,representative_channel_rate*station_channel_limit)
                planning_demand=cfg.traffic_total_gbps*cfg.traffic_landing_percent/100.0
                if traffic_cache_connection is not None:
                    demand_samples=sorted({round(i*(total_frames-1)/max(1,min(61,total_frames)-1)) for i in range(min(61,total_frames))})
                    traffic_source=traffic_cache_blobs if traffic_cache_blobs is not None else traffic_cache_connection
                    planning_demand=max((sum(float(row[1]) for row in read_traffic_cache_frame(traffic_source,index)[1]) for index in demand_samples),default=cfg.traffic_total_gbps)*cfg.traffic_landing_percent/100.0
                capacity_target=planning_demand*(1.0+max(0.0,cfg.planning_capacity_reserve_percent)/100.0)
                gateway_spec={"cache_version":12,"algorithm_version":14,"orbit_key":cache_key,"traffic_key":traffic_key,
                    "orbit":{"source":cfg.source,"shells":cfg.walker_shells,"count":len(satellites),"planes":cfg.planes,
                        "layout":cfg.walker_layout,"altitude_km":cfg.altitude_km,"inclination_deg":cfg.inclination_deg,
                        "eccentricity":cfg.eccentricity,"arg_perigee_deg":cfg.arg_perigee_deg,"duration_s":cfg.duration_s,"step_s":cfg.step_s},
                    "mode":cfg.gateway_planning_mode,"specified_stations":cfg.stations if cfg.gateway_planning_mode=="specified" else [],
                    "min_elevation_deg":cfg.min_elevation_deg,"max_gsl_km":cfg.max_gsl_km,
                    "minimum_sites":cfg.gateway_min_sites,"maximum_sites":cfg.gateway_max_sites,"redundancy":cfg.gateway_redundancy,"maximum_antennas":cfg.gateway_max_antennas,
                    "feeder_bandwidth_mhz":cfg.feeder_bandwidth_mhz,"feeder_frequency_ghz":cfg.feeder_frequency_ghz,"feeder_rolloff":cfg.feeder_rolloff,"satellite_eirp_dbw":cfg.satellite_eirp_dbw,"gateway_gt_dbk":cfg.gateway_gt_dbk,"gateway_array_loss_db":cfg.gateway_array_loss_db,"snr_margin_db":cfg.snr_margin_db,"modcod_table":[[row.name,row.minimum_snr_db,row.spectral_efficiency,row.rate_gbps] for row in modcod_table],
                    "geo_interference":{"enabled":cfg.geo_interference_enabled,"satellite_count":cfg.geo_satellite_count,"center_longitude_deg":cfg.geo_center_longitude_deg,
                        "longitude_spacing_deg":cfg.geo_longitude_spacing_deg,"altitude_km":cfg.geo_altitude_km,"i_over_n_db":cfg.gso_i_over_n_db,"protection_angle_deg":cfg.gso_protection_angle_deg},
                    "traffic":{"enabled":cfg.traffic_enabled,"grid_hash":traffic_source_signature,"outside_china_percent":cfg.traffic_outside_china_percent,
                        "total_gbps":cfg.traffic_total_gbps,"dense_percentile":cfg.traffic_dense_percentile,"dense_max_satellites":cfg.traffic_dense_max_satellites,
                        "random_sigma":cfg.traffic_random_sigma,"cluster_radius_km":cfg.traffic_cluster_radius_km},
                    "traffic_landing_percent":cfg.traffic_landing_percent,"planning_demand_gbps":round(planning_demand,9),
                    "planning_capacity_reserve_percent":round(cfg.planning_capacity_reserve_percent,6),
                    "planning_capacity_target_gbps":round(capacity_target,9),
                    "rain":{"enabled":cfg.rain_enabled,"sense_sigma":cfg.rain_sense_sigma,
                        "forecast_hours":cfg.rain_forecast_hours,"use_p90":cfg.rain_use_p90},
                    "effective_station_capacity_gbps":round(effective_station_capacity,9),"traffic_cell_count":len(traffic_cells)}
                gateway_key=hashlib.sha256(json.dumps(gateway_spec,sort_keys=True,ensure_ascii=False,separators=(",",":")).encode("utf-8")).hexdigest()[:24]
                gateway_root=cache_root/"gateways";gateway_root.mkdir(parents=True,exist_ok=True)
                deployment_path=gateway_root/f"{gateway_key}.json"
                if deployment_path and deployment_path.is_file():
                    try:
                        cached_deployment=json.loads(deployment_path.read_text(encoding="utf-8"))
                        if cached_deployment.get("cache_spec")==gateway_spec and cached_deployment.get("planned_stations"):
                            gateway_deployment=cached_deployment;gateway_cache_reused=True
                    except (OSError,ValueError,TypeError):
                        gateway_cache_reused=False
                if gateway_cache_reused:
                    self.progress.emit(98,"准备阶段 4/4：已复用场景信关站规划缓存")
                else:
                    self.progress.emit(98,"准备阶段 4/4：正在计算固定信关站资源规划")
                    if cfg.gateway_planning_mode=="automatic":
                        gateway_deployment=plan_gateway_automatic(orbit_cache,cfg.min_elevation_deg,effective_station_capacity,
                            capacity_target,cfg.max_gsl_km,traffic_cells,cfg.gateway_max_sites,cfg.gateway_redundancy,cfg.gateway_max_antennas,
                            cfg.rain_enabled,cfg.rain_sense_sigma,cfg.rain_forecast_hours,cfg.rain_use_p90,
                            planner.config if planner is not None else None,minimum_sites=cfg.gateway_min_sites)
                    else:
                        gateway_deployment=plan_gateway_resources(cfg.stations,orbit_cache,cfg.min_elevation_deg,
                            effective_station_capacity,capacity_target,cfg.max_gsl_km,traffic_cells,
                            cfg.rain_enabled,cfg.rain_sense_sigma,cfg.rain_forecast_hours,cfg.rain_use_p90,
                            planner.config if planner is not None else None)
                    for station_metric in gateway_deployment.get("stations",[]):
                        share=float(station_metric.get("traffic_share",0.0))
                        target_share=capacity_target*share;service_share=planning_demand*share
                        station_metric["assigned_demand_gbps"]=round(service_share,3)
                        station_metric["capacity_target_gbps"]=round(target_share,3)
                        station_metric["reserve_capacity_gbps"]=round(max(0.0,target_share-service_share),3)
                    gateway_deployment.update(effective_station_radio_capacity_gbps=round(effective_station_capacity,3),
                        representative_channel_rate_gbps=round(representative_channel_rate,3),radio_availability_factor=1.00,
                        service_demand_gbps=round(planning_demand,3),capacity_reserve_percent=round(cfg.planning_capacity_reserve_percent,3),
                        capacity_target_gbps=round(capacity_target,3),
                        cache_spec=gateway_spec,cache_created_at=datetime.now(timezone.utc).isoformat())
                planned_stations=gateway_deployment.get("planned_stations") or []
                if not planned_stations:raise RuntimeError("信关站资源规划失败，请检查预配置站点")
                cfg.stations=planned_stations
                if planner is not None:planner.stations=list(planned_stations)
                if deployment_path and not gateway_cache_reused:
                    deployment_path.parent.mkdir(parents=True,exist_ok=True);temporary_deployment=deployment_path.with_suffix(".building.json")
                    temporary_deployment.write_text(json.dumps(gateway_deployment,ensure_ascii=False,indent=2),encoding="utf-8");temporary_deployment.replace(deployment_path)
            rain_key="";rain_cache_path="";rain_spatial_cache_reused=False
            rain_model_key="";rain_model_path="";rain_cache_reused=False
            install_scenario_spatial_cache(())
            if cfg.rain_enabled:
                rain_model_spec={"cache_version":1,"source":cfg.rain_source,
                                 "sense_sigma":cfg.rain_sense_sigma,"forecast_hours":cfg.rain_forecast_hours,
                                 "use_p90":cfg.rain_use_p90,"duration_s":cfg.duration_s,"step_s":cfg.step_s,
                                 "stations":[{key:site.get(key) for key in ("lat","lon","alt","frequency_ghz",
                                              "polarization_tilt_deg","rain_rate_001_mmh","rain_rate_mmh",
                                              "representative_rain_rate_mmh") if key in site} for site in cfg.stations]}
                rain_model_key=hashlib.sha256(json.dumps(rain_model_spec,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode("utf-8")).hexdigest()[:24]
                rain_root=cache_root/"rain";rain_root.mkdir(parents=True,exist_ok=True)
                model_file=rain_root/f"model_{rain_model_key}.json";rain_model_path=str(model_file)
                if model_file.is_file():
                    try:rain_cache_reused=json.loads(model_file.read_text(encoding="utf-8")).get("spec")==rain_model_spec
                    except (OSError,ValueError,TypeError):pass
                if not rain_cache_reused:
                    temporary_model=model_file.with_suffix(".building.json")
                    temporary_model.write_text(json.dumps({"spec":rain_model_spec},ensure_ascii=False,separators=(",",":")),encoding="utf-8")
                    temporary_model.replace(model_file)
                rain_slots=range(max(1,int(math.floor(cfg.duration_s/300.))+1))
                rain_spec={"cache_version":1,"profile":RAIN_GRADE_SPATIAL_PROFILES,
                           "stations":[[round(float(site.get("lat",0)),6),round(float(site.get("lon",0)),6)] for site in cfg.stations],
                           "slot_count":len(rain_slots)}
                rain_key=hashlib.sha256(json.dumps(rain_spec,sort_keys=True,separators=(",",":")).encode("utf-8")).hexdigest()[:24]
                rain_file=rain_root/f"{rain_key}.json";rain_cache_path=str(rain_file);rain_rows=None
                if rain_file.is_file():
                    try:
                        stored=json.loads(rain_file.read_text(encoding="utf-8"))
                        if stored.get("spec")==rain_spec and len(stored.get("factors",[]))==len(cfg.stations)*len(RAIN_GRADE_SPATIAL_PROFILES)*len(rain_slots):
                            rain_rows=stored["factors"];rain_spatial_cache_reused=True
                    except (OSError,ValueError,KeyError,TypeError):pass
                if rain_rows is None:
                    rain_rows=[[float(site.get("lat",0)),float(site.get("lon",0)),grade,slot,
                                scenario_spatial_factor(site,grade,slot*300.)]
                               for site in cfg.stations for grade in RAIN_GRADE_SPATIAL_PROFILES for slot in rain_slots]
                    temporary_rain=rain_file.with_suffix(".building.json")
                    temporary_rain.write_text(json.dumps({"spec":rain_spec,"factors":rain_rows},ensure_ascii=False,separators=(",",":")),encoding="utf-8")
                    temporary_rain.replace(rain_file)
                install_scenario_spatial_cache(rain_rows)
            geo_satellites=[]
            if cfg.geo_interference_enabled:
                offset=(cfg.geo_satellite_count-1)/2
                geo_satellites=[{"id":index,"name":f"GEO-{index+1:02d}","longitude_deg":round(cfg.geo_center_longitude_deg+(index-offset)*cfg.geo_longitude_spacing_deg,4),"altitude_km":cfg.geo_altitude_km} for index in range(cfg.geo_satellite_count)]
            aggregate_i_over_n_db=(cfg.gso_i_over_n_db+10*math.log10(max(1,cfg.geo_satellite_count))) if cfg.geo_interference_enabled else None
            gso_penalty_db=10*math.log10(1+10**(aggregate_i_over_n_db/10)) if aggregate_i_over_n_db is not None else 0.0
            scenario_basic=Path(cfg.scenario_directory)/"base_data"
            scenario_basic.mkdir(parents=True,exist_ok=True)
            asset_manifest={"asset_type":"basic_data_references","cache_version":1,
                "tle":{"key":tle_asset_hash[:24],"path":str(shared_tle_asset),"scenario_file":str(tle_asset)},
                "topology":{"key":topology_key,"path":str(topology_path),"reused":topology_cache_reused},
                "orbit":{"key":cache_key,"path":orbit_cache_path,"reused":orbit_cache_reused},
                "traffic":{"key":traffic_key,"path":traffic_cache_path,"reused":traffic_cache_reused},
                "gateways":{"key":gateway_key,"path":str(deployment_path) if deployment_path else "","reused":gateway_cache_reused}}
            asset_manifest["rain"]={"key":rain_model_key,"path":rain_model_path,"reused":rain_cache_reused,
                                    "spatial_key":rain_key,"spatial_path":rain_cache_path,"spatial_reused":rain_spatial_cache_reused}
            asset_manifest["isl_model"]={"key":isl_model_key,"path":str(isl_model_file),"reused":isl_model_reused}
            (scenario_basic/"basic_assets_manifest.json").write_text(json.dumps(asset_manifest,ensure_ascii=False,indent=2),encoding="utf-8")
            self.progress.emit(98, f"准备阶段 4/4：正在建立 {total_frames:,} 个时间点的时间轴和场景描述")
            self.prepared.emit({"satellite_count": len(satellites), "planes": cfg.planes,
                "per_plane": per_plane, "frames": total_frames, "epoch": epoch.isoformat(),
                "walker_mode": cfg.walker_mode, "walker_shells": shell_runs,
                "orbit_cache": bool(orbit_cache is not None), "orbit_cache_path": orbit_cache_path,"orbit_cache_reused":orbit_cache_reused,
                "topology_cache_path":str(topology_path),"topology_cache_reused":topology_cache_reused,
                "isl_model_path":str(isl_model_file),"isl_model_reused":isl_model_reused,
                "orbit_cache_size_mb": round(orbit_cache_bytes/1024/1024,1) if orbit_cache is not None else 0,
                "traffic_cache": bool(traffic_cache_connection is not None),"traffic_cache_path":traffic_cache_path,"traffic_cache_reused":traffic_cache_reused,
                "gateway_cache_reused":gateway_cache_reused,
                "rain_cache_path":rain_cache_path,"rain_cache_reused":rain_cache_reused,
                "rain_model_path":rain_model_path,"rain_spatial_cache_reused":rain_spatial_cache_reused,
                "geo_interference_enabled":cfg.geo_interference_enabled,"geo_satellites":geo_satellites,
                "gso_i_over_n_db":cfg.gso_i_over_n_db,"gso_aggregate_i_over_n_db":round(aggregate_i_over_n_db,3) if aggregate_i_over_n_db is not None else None,
                "gso_interference_penalty_db":round(gso_penalty_db,3),"gso_protection_angle_deg":cfg.gso_protection_angle_deg,
                "traffic_cache_preloaded":bool(traffic_cache_blobs is not None),"traffic_cache_compressed_mb":round(compressed_bytes/1024/1024,1),
                "traffic_update_period_s":traffic_update_period_s,"traffic_keyframe_count":traffic_keyframe_count,
                "satellite_labels": satellite_labels,
                "tle_asset_path":str(tle_asset),"tle_asset_hash":tle_asset_hash,
                "constellation_cache_reused":constellation_cache_reused if cfg.source=="walker" else False,
                "step_s": cfg.step_s,
                "latitude_limit": cfg.latitude_limit_deg, "stations": cfg.stations,
                "min_elevation_deg": cfg.min_elevation_deg, "max_gsl_km": cfg.max_gsl_km,
                "isl_pattern": cfg.isl_pattern,
                "isl_candidate_count": len(candidates),
                "seam_candidate_count": len(seam_candidates),
                "walker_layout": cfg.walker_layout,
                "seam_enabled": cfg.seam_enabled, "seam_after_plane": cfg.seam_after_plane,
                "center_plane": cfg.center_plane, "display_plane_count": cfg.display_plane_count,
                "perigee_altitude_km": perigee_altitude_km if cfg.source == "walker" else None,
                "apogee_altitude_km": apogee_altitude_km if cfg.source == "walker" else None,
                "traffic_enabled": cfg.traffic_enabled,
                "traffic_cell_count": len(traffic_cells),
                "traffic_cells": [[cell.cell_id,round(cell.latitude_deg,6),round(cell.longitude_deg,6)] for cell in traffic_cells],
                "traffic_outside_china_percent": cfg.traffic_outside_china_percent,
                "traffic_total_gbps": cfg.traffic_total_gbps,
                "traffic_cluster_radius_km": cfg.traffic_cluster_radius_km,
                "planning_enabled": cfg.planning_enabled,
                "gateway_deployment":gateway_deployment,
                "planning_period_s": cfg.planning_period_s,
                "planning_horizon_s":cfg.planning_horizon_s,
                "planning_slot_s":cfg.planning_slot_s,
                "fast_evaluation_period_s":cfg.fast_evaluation_period_s,
                "fast_adjust_period_s":cfg.fast_adjust_period_s,
                "route_update_period_s":cfg.route_update_period_s,
                "planning_freeze_window_s":cfg.planning_freeze_window_s,
                "feeder_bandwidth_mhz": cfg.feeder_bandwidth_mhz,"feeder_frequency_ghz":cfg.feeder_frequency_ghz,"feeder_rolloff":cfg.feeder_rolloff,"satellite_eirp_dbw":cfg.satellite_eirp_dbw,
                "snr_margin_db": cfg.snr_margin_db,
                "capacity_threshold_percent": cfg.capacity_threshold_percent,
                "switching_loss_percent": cfg.switching_loss_percent,
                "modcod_table": [[row.name, row.minimum_snr_db, row.spectral_efficiency, row.rate_gbps]
                                  for row in modcod_table],
                "layers": {"satellites": cfg.show_satellites, "same": cfg.show_same_isl,
                           "cross": cfg.show_cross_isl, "gsl": cfg.show_gsl,
                           "motion": cfg.show_motion, "labels": cfg.show_labels}})
            self.progress.emit(100, f"准备完成：{len(satellites):,}颗卫星，{len(candidates):,}条ISL候选，{total_frames:,}个时间点；即将自动播放")
            observer = ephem.Observer(); observer.lat = "0"; observer.lon = "0"; observer.elevation = 0
            executor=ThreadPoolExecutor(max_workers=3,thread_name_prefix="frame-calc")
            next_frame=0;next_due=time.perf_counter();first_frame_submitted=False;gc_was_enabled=gc.isenabled()
            while not self._stop.is_set():
                with self._control_lock:
                    requested = self._requested_frame
                    if requested is not None:
                        self._requested_frame = None
                        next_frame = max(0, min(total_frames - 1, requested))
                        next_due=time.perf_counter()
                    paused = self._paused
                    strategy_request=self._planning_strategy_request
                    self._planning_strategy_request=None
                    selection_request=self._planning_selection_request
                    self._planning_selection_request=None
                    reserve_request=self._planning_reserve_request
                    self._planning_reserve_request=None
                    scheduling_request=self._scheduling_request
                    self._scheduling_request=None
                    offline_requested=self._offline_plan_requested
                    self._offline_plan_requested=False
                    offline_grade=self._offline_grade
                if strategy_request is not None and planner is not None:
                    planner.config.capacity_weight,planner.config.delay_weight=strategy_request
                    planner.state.plans=[];planner.state.execution_plan=None;planner.state.last_result=None
                    planner.state.generated_at_s=-math.inf
                    append_diagnostic(diagnostic_path,"planning_strategy_applied",frame=next_frame,
                        capacity_weight=round(strategy_request[0],6),delay_weight=round(strategy_request[1],6),
                        delay_reference_ms=round(planner.config.delay_reference_ms,6),forces_replan=True)
                if selection_request is not None and planner is not None:
                    planner.config.selection_strategy=selection_request
                    planner.state.plans=[];planner.state.execution_plan=None;planner.state.last_result=None
                    planner.state.generated_at_s=-math.inf
                if reserve_request is not None and planner is not None:planner.config.capacity_reserve_ratio=reserve_request
                if scheduling_request is not None and planner is not None:
                    planner.config.fast_evaluation_period_s,planner.config.fast_adjust_period_s,planner.config.route_update_period_s,planner.config.freeze_window_s,planner.config.capacity_threshold_ratio,planner.config.switching_loss_ratio=scheduling_request
                if offline_requested and planner is not None:
                    if self._stop.is_set():return
                    grades=(("轻雨",5.0),("中雨",1.0),("重雨",0.1));catalog=[];planning_interval_s=300
                    offline_spec={"cache_version":13,"algorithm_version":29,"candidate_label_cost":"isl_plus_gsl_v1","max_isl_km":cfg.max_isl_km,"latitude_limit_deg":cfg.latitude_limit_deg,"isl_pattern":cfg.isl_pattern,"seam_enabled":cfg.seam_enabled,"seam_after_plane":cfg.seam_after_plane,"fixed_feeder_visibility":"sampled_full_300s_v1","planning_traffic_sampling":"six_samples_mean_spatial_peak_total_v1","capacity_safe_latency":"local_self_first_v2","delay_score":"smooth_15ms_v1","planning_exit_limit":8,"objective_formula":"capacity_then_delay_lexicographic_v1","delay_diversity":"none","gateway_channel_capacity_gbps":2.0,"feeder_selection":"capacity_first_spatial_diversity_v1","feeder_min_separation_deg":8.0,"feeder_diversity_target_deg":25.0,"rain_spatial_model":"correlated_cells_v1","joint_link_repair":"delivery_first_single_pass_v1","orbit_key":cache_key,"traffic_key":traffic_key,
                        "gateway_cache_version":(gateway_deployment.get("cache_spec") or {}).get("cache_version",0),"station_signature":[[s.get("name"),s.get("lat"),s.get("lon"),s.get("channel_count"),s.get("antenna_count")] for s in cfg.stations],
                        "duration_s":1800,"planning_interval_s":planning_interval_s,"rain_grade_percentiles":grades,
                        "capacity_weight":round(planner.config.capacity_weight,6),"delay_weight":round(planner.config.delay_weight,6),
                        "capacity_reserve_ratio":round(planner.config.capacity_reserve_ratio,6),"traffic_landing_ratio":round(planner.config.traffic_landing_ratio,6),
                        "bandwidth_mhz":planner.config.bandwidth_mhz,"snr_margin_db":planner.config.snr_margin_db,"frequency_ghz":planner.config.feeder_frequency_ghz,"rolloff":planner.config.feeder_rolloff,"satellite_eirp_dbw":planner.config.satellite_eirp_dbw,"gateway_gt_dbk":planner.config.gateway_gt_dbk,"gateway_array_loss_db":planner.config.gateway_array_loss_db,
                        "modcod":[[row.name,row.minimum_snr_db,row.spectral_efficiency,row.rate_gbps] for row in planner.config.modcod_table],
                        "priority_regions":list(planner.config.priority_task_regions),
                        "gso":[planner.config.geo_interference_enabled,planner.config.geo_satellite_count,planner.config.gso_i_over_n_db,planner.config.gso_protection_angle_deg],
                        "priority_routing_version":4}
                    offline_spec["tle_asset_hash"]=tle_asset_hash
                    offline_spec.update(topology_key=topology_key,rain_spatial_key=rain_key,min_elevation_deg=cfg.min_elevation_deg,max_gsl_km=cfg.max_gsl_km,
                        rain_model=[cfg.rain_enabled,cfg.rain_source,cfg.rain_sense_sigma,cfg.rain_forecast_hours,cfg.rain_use_p90],
                        geo_geometry=[cfg.geo_center_longitude_deg,cfg.geo_longitude_spacing_deg,cfg.geo_altitude_km])
                    if planner.config.selection_strategy=="window_margin":offline_spec.update(cache_version=16,algorithm_version=32,objective_formula="weighted_window_mean_capacity_delay_v4",selection_strategy="window_margin",window_sampling="six_points_mean_and_min_rate_v1")
                    offline_spec_hash=hashlib.sha256(json.dumps(offline_spec,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode("utf-8")).hexdigest()
                    planning_root=cache_root/"planning";planning_root.mkdir(parents=True,exist_ok=True)
                    offline_cache_path=planning_root/f"{offline_spec_hash[:24]}.json"
                    cached_statistics={}
                    def record_offline_asset(cache_reused,statistics):
                        if not offline_cache_path:return
                        scenario_basic=Path(cfg.scenario_directory)/"base_data" if cfg.scenario_directory else offline_cache_path.parent
                        scenario_basic.mkdir(parents=True,exist_ok=True)
                        scenario_copy=scenario_basic/"offline_plans.json"
                        if scenario_copy!=offline_cache_path:shutil.copy2(offline_cache_path,scenario_copy)
                        manifest_path=scenario_basic/"basic_assets_manifest.json"
                        try:manifest=json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
                        except (OSError,ValueError,TypeError):manifest={}
                        manifest["offline_planning"]={"file":scenario_copy.name,"shared_cache_path":str(offline_cache_path),"planning_spec_hash":offline_spec_hash,
                            "plan_count":len(catalog),"cache_reused":bool(cache_reused),**statistics}
                        manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
                    if offline_cache_path and offline_cache_path.is_file():
                        try:
                            cached_offline=json.loads(offline_cache_path.read_text(encoding="utf-8"))
                            cached_spec=cached_offline.get("spec") or {}
                            cached_spec_hash=cached_offline.get("planning_spec_hash") or hashlib.sha256(json.dumps(cached_spec,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode("utf-8")).hexdigest()
                            if cached_spec_hash==offline_spec_hash and len(cached_offline.get("plans",[]))==18:
                                catalog=cached_offline["plans"];cached_statistics=cached_offline.get("statistics") or {}
                        except (OSError,ValueError,TypeError):catalog=[]
                    if catalog:
                        if self._stop.is_set():return
                        batch_total_ms=float(cached_statistics.get("total_calculation_ms",catalog[0].get("batch_total_calculation_ms",sum(float(p.get("calculation_ms",0)) for p in catalog))))
                        for plan in catalog:plan["batch_total_calculation_ms"]=batch_total_ms;plan["planning_cache_reused"]=True;plan["planning_spec_hash"]=offline_spec_hash
                        record_offline_asset(True,{"total_calculation_ms":batch_total_ms,"average_calculation_ms":float(cached_statistics.get("average_calculation_ms",0)),"maximum_calculation_ms":float(cached_statistics.get("maximum_calculation_ms",0))})
                        self._offline_catalog=catalog;self._installed_offline_key=None
                        if self.result_store is not None:self.result_store.set_metadata("offline_plans",catalog)
                        self.offline_planning_ready.emit(catalog);self.progress.emit(100,"离线规划：参数未变化，已直接加载18个缓存方案")
                        append_diagnostic(diagnostic_path,"offline_planning_cache_loaded",plan_count=len(catalog),planning_interval_s=planning_interval_s)
                        continue
                    planning_started=time.perf_counter();self.progress.emit(1,"离线规划：正在生成18个规划方案")
                    for epoch_index,plan_time in enumerate(range(0,1800,planning_interval_s)):
                        if self._stop.is_set():return
                        frame_at=min(total_frames-1,max(0,round(plan_time/cfg.step_s)))
                        cached=np.asarray(orbit_cache[frame_at]);count=len(satellites)
                        position_array=np.empty((count,5),dtype=np.float64);position_array[:,0]=np.arange(count);position_array[:,1]=satellite_planes;position_array[:,2:4]=np.round(cached[:,0:2],6);position_array[:,4]=np.round(cached[:,2],1)
                        plan_positions=position_array.tolist()
                        for row in plan_positions:row[0]=int(row[0]);row[1]=int(row[1])
                        coordinates=np.asarray(cached[:,3:6]);edge_array=candidate_array;a_idx=edge_array[:,0];b_idx=edge_array[:,1];kinds=edge_array[:,2];point_a=coordinates[a_idx];delta=coordinates[b_idx]-point_a;distance_sq=np.einsum("ij,ij->i",delta,delta)
                        valid=np.ones(len(edge_array),dtype=bool)
                        if cfg.max_isl_km>0:valid&=distance_sq<=(cfg.max_isl_km*1000)**2
                        latitude=cached[:,0];valid&=(kinds==0)|((np.abs(latitude[a_idx])<=cfg.latitude_limit_deg)&(np.abs(latitude[b_idx])<=cfg.latitude_limit_deg))
                        denominator=np.maximum(distance_sq,1.0);closest_t=np.clip(-np.einsum("ij,ij->i",point_a,delta)/denominator,0.0,1.0);closest=point_a+closest_t[:,None]*delta;valid&=np.einsum("ij,ij->i",closest,closest)>EARTH_RADIUS_M**2
                        plan_links=edge_array[valid].tolist();plan_gsls,_=compute_gsl_rows(coordinates,cfg.stations,cfg.min_elevation_deg,cfg.max_gsl_km)
                        # The fixed feeder set must remain geometrically usable
                        # across this entire five-minute planning interval.
                        stable={(int(g),int(s)):float(e) for g,s,e in plan_gsls}
                        profiles={pair:[(plan_time,elevation,float(cached[pair[1],2])/1000.)]
                                  for pair,elevation in stable.items()} if planner.config.selection_strategy=="window_margin" else None
                        sample_times=[min(plan_time+offset,1799) for offset in (60,120,180,240,299)]
                        for sample_time in sample_times:
                            if self._stop.is_set():return
                            sample_index=min(total_frames-1,max(0,round(sample_time/cfg.step_s)))
                            sample_cached=np.asarray(orbit_cache[sample_index]);sample_xyz=sample_cached[:,3:6]
                            sample_rows,_=compute_gsl_rows(sample_xyz,cfg.stations,cfg.min_elevation_deg,cfg.max_gsl_km)
                            visible={(int(g),int(s)):float(e) for g,s,e in sample_rows}
                            stable={pair:min(elevation,visible[pair]) for pair,elevation in stable.items() if pair in visible}
                            if profiles is not None:
                                profiles={pair:profiles[pair]+[(sample_time,visible[pair],float(sample_cached[pair[1],2])/1000.)]
                                          for pair in stable}
                        plan_gsls=[[gateway,satellite,round(elevation,2)] for (gateway,satellite),elevation in stable.items()]
                        traffic_source=traffic_cache_blobs if traffic_cache_blobs is not None else traffic_cache_connection
                        _traffic,plan_traffic,_elapsed=read_traffic_cache_frame(traffic_source,frame_at)
                        # Route selection sees the entire fixed five-minute
                        # window.  Average per-source spatial demand across
                        # six samples, then scale to the peak sampled total.
                        # This avoids treating all six peaks as simultaneous.
                        traffic_sum={};priority_sum={};sample_totals=[]
                        for sample_index in [frame_at]+[min(total_frames-1,max(0,round(ti/cfg.step_s))) for ti in sample_times]:
                            if sample_index==frame_at:
                                source_rows=plan_traffic;grid_rows=_traffic
                            else:
                                grid_rows,source_rows,_sample_elapsed=read_traffic_cache_frame(traffic_source,sample_index)
                            for sid,amount in priority_traffic_by_satellite(grid_rows,cfg.priority_task_regions).items():
                                priority_sum[sid]=priority_sum.get(sid,0.)+amount
                            sample_totals.append(sum(float(row[1]) for row in source_rows))
                            for row in source_rows:
                                sid=int(row[0]);traffic_sum[sid]=traffic_sum.get(sid,0.)+float(row[1])
                        average_total=sum(sample_totals)/len(sample_totals)
                        scale=max(sample_totals)/average_total if average_total>1e-12 else 1.
                        plan_traffic=[[sid,amount*scale/len(sample_totals)] for sid,amount in traffic_sum.items() if amount>1e-12]
                        plan_priority={sid:amount*scale/len(sample_totals) for sid,amount in priority_sum.items() if amount>1e-12}
                        for grade,scenario_percentile in grades:
                            if self._stop.is_set():return
                            plan_started=time.perf_counter()
                            window_rates=None
                            if profiles is not None:
                                window_rates={}
                                for (gateway,satellite),samples in profiles.items():
                                    station=cfg.stations[gateway];states=[]
                                    for sample_time,elevation,altitude_km in samples:
                                        fade=scenario_attenuation_db(station,elevation,grade,sample_time) if planner.config.rain_enabled else 0.
                                        slant=_slant_distance_km(altitude_km,elevation)
                                        snr,mcs,rate=link_state(elevation,planner.config,fade,slant)
                                        states.append((rate,snr,mcs,fade,elevation,sample_time))
                                    rate,snr,mcs,fade,elevation,worst_time=min(states,key=lambda item:(item[0],item[1]))
                                    window_rates[(gateway,satellite)]={"rate_gbps":rate,"mean_rate_gbps":sum(item[0] for item in states)/len(states),"snr_db":snr,"modcod":mcs,
                                                                         "rain_attenuation_db":fade,"worst_elevation_deg":elevation,
                                                                         "worst_time_s":worst_time}
                            plan=planner.generate(plan_time,plan_positions,plan_links,plan_gsls,plan_traffic,rain_scenario_percentile=scenario_percentile,rain_grade=grade,commit=False,window_rates=window_rates,priority_traffic=plan_priority)
                            if self._stop.is_set():return
                            plan.update(id=f"offline_{grade}_{plan_time}",generation=len(catalog)+1,offline=True,epoch_index=epoch_index,
                                        valid_from_s=float(plan_time),valid_until_s=float(plan_time+planning_interval_s),calculation_ms=round((time.perf_counter()-plan_started)*1000,3),
                                        full_window_visibility_required=True,visibility_sample_times_s=[plan_time]+sample_times,
                                        stable_visible_candidate_count=len(plan_gsls),tle_asset="base_data/satellites.tles",tle_asset_hash=tle_asset_hash)
                            for link in plan.get("links",[]):
                                name,line1,line2=tle_records[int(link["satellite"])]
                                link["tle"]={"name":name,"line1":line1,"line2":line2}
                            catalog.append(plan)
                            self.progress.emit(round(100*len(catalog)/18),f"离线规划：已完成 {len(catalog)}/18，{plan_time//60:02d}:00 {grade}，本方案 {plan['calculation_ms']/1000:.1f}s")
                    self._offline_catalog=catalog;self._installed_offline_key=None
                    if self._stop.is_set():return
                    total_planning_ms=round((time.perf_counter()-planning_started)*1000,3)
                    statistics={"total_calculation_ms":total_planning_ms,"average_calculation_ms":round(sum(float(p.get("calculation_ms",0)) for p in catalog)/max(1,len(catalog)),3),"maximum_calculation_ms":max((p.get("calculation_ms",0) for p in catalog),default=0),"generated_at":datetime.now().astimezone().isoformat()}
                    for plan in catalog:plan["batch_total_calculation_ms"]=total_planning_ms;plan["planning_cache_reused"]=False;plan["planning_spec_hash"]=offline_spec_hash
                    if offline_cache_path:
                        offline_cache_path.parent.mkdir(parents=True,exist_ok=True);temporary_offline=offline_cache_path.with_suffix(".building.json")
                        temporary_offline.write_text(json.dumps({"spec":offline_spec,"planning_spec_hash":offline_spec_hash,"statistics":statistics,"plans":catalog},ensure_ascii=False,separators=(",",":")),encoding="utf-8");temporary_offline.replace(offline_cache_path)
                        record_offline_asset(False,statistics)
                    if self.result_store is not None:self.result_store.set_metadata("offline_plans",catalog)
                    append_diagnostic(diagnostic_path,"offline_planning_completed",plan_count=len(catalog),total_ms=total_planning_ms,average_ms=round(total_planning_ms/max(1,len(catalog)),3),maximum_ms=max((p.get("calculation_ms",0) for p in catalog),default=0))
                    self.offline_planning_ready.emit(catalog);self.progress.emit(100,f"离线规划完成：18个方案，用时 {total_planning_ms/1000:.1f}s")
                    continue
                if paused and requested is None:
                    if gc_was_enabled and not gc.isenabled():gc.enable()
                    next_due=time.perf_counter();self._stop.wait(0.03);continue
                frame_index = next_frame
                if self._stop.is_set(): break
                # A frame creates tens of thousands of short-lived lists. The
                # cyclic collector occasionally scanned them in the middle of
                # NumPy/graph stages, producing 0.6-1.1 s random stalls. Refcount
                # cleanup remains active; defer cyclic collection until pause/end.
                if gc.isenabled():gc.disable()
                wall_start = time.perf_counter(); sim_time = min(frame_index * cfg.step_s, cfg.duration_s)
                if planner is not None and self._offline_catalog:
                    epoch_start=min(1500,int(sim_time//300)*300);key=(offline_grade,epoch_start)
                    if key!=self._installed_offline_key:
                        selected=next((row for row in self._offline_catalog if row.get("rain_grade")==offline_grade and int(row.get("valid_from_s",0))==epoch_start),None)
                        if selected is not None:planner.install_offline_plan(selected);self._installed_offline_key=key
                stage_start=wall_start;timings={}
                observer.date = epoch + timedelta(seconds=sim_time)
                positions, xyz = [], []
                if orbit_cache is not None:
                    # Read the mapped frame as one contiguous block. Iterating a
                    # memmap row-by-row creates NumPy scalar objects and was the
                    # dominant cost for 20,000 satellites.
                    cached=np.asarray(orbit_cache[frame_index]);count=len(satellites)
                    position_array=np.empty((count,5),dtype=np.float64);position_array[:,0]=np.arange(count);position_array[:,1]=satellite_planes;position_array[:,2:4]=np.round(cached[:,0:2],6);position_array[:,4]=np.round(cached[:,2],1)
                    positions=position_array.tolist()
                    for row in positions:row[0]=int(row[0]);row[1]=int(row[1])
                    xyz=cached[:,3:6]
                elif analytic_walker:
                    try:
                        import numpy as np
                        params=np.asarray(analytic_params,dtype=float);radius=params[:,0];inc=params[:,1];raan=params[:,2];u=params[:,3]+params[:,4]*sim_time
                        cos_u=np.cos(u);sin_u=np.sin(u);cos_o=np.cos(raan);sin_o=np.sin(raan);cos_i=np.cos(inc);sin_i=np.sin(inc)
                        x=radius*(cos_o*cos_u-sin_o*sin_u*cos_i);y=radius*(sin_o*cos_u+cos_o*sin_u*cos_i);z=radius*sin_u*sin_i
                        earth_angle=greenwich_sidereal_angle(epoch)+7.2921150e-5*sim_time;ce=math.cos(earth_angle);se=math.sin(earth_angle);xe=ce*x+se*y;ye=-se*x+ce*y
                        latitudes=np.degrees(np.arctan2(z,np.sqrt(xe*xe+ye*ye)));longitudes=np.degrees(np.arctan2(ye,xe));altitudes=radius-EARTH_RADIUS_M
                        xyz=list(zip(xe.tolist(),ye.tolist(),z.tolist()))
                        positions=[[sid,satellite_planes[sid],round(float(latitudes[sid]),6),round(float(longitudes[sid]),6),round(float(altitudes[sid]),1)] for sid in range(len(satellites))]
                    except ImportError as exc:
                        raise RuntimeError("大规模Walker快速计算需要numpy") from exc
                else:
                    for sid, satellite in enumerate(satellites):
                        try:satellite.compute(observer)
                        except RuntimeError as exc:raise RuntimeError(f"卫星 {sid} 无法在 t={sim_time:g}s（{observer.date}）进行轨道外推。请检查TLE有效性，以及椭圆轨道的近地点是否进入地球或大气层。") from exc
                        lat=math.degrees(float(satellite.sublat));lon=math.degrees(float(satellite.sublong));alt=float(satellite.elevation)
                        positions.append([sid,satellite_planes[sid],round(lat,6),round(lon,6),round(alt,1)]);xyz.append(geodetic_to_ecef(lat,lon,alt))
                timings["orbit_ms"]=round((time.perf_counter()-stage_start)*1000,1);stage_start=time.perf_counter()
                frame_time=epoch+timedelta(seconds=sim_time)
                gsl_future=executor.submit(compute_gsl_rows,xyz,cfg.stations,cfg.min_elevation_deg,cfg.max_gsl_km)
                traffic_cache_source=traffic_cache_blobs if traffic_cache_blobs is not None else traffic_cache_connection
                if traffic_cache_blobs is not None:
                    # Decode in the simulation thread. Background zlib/marshal
                    # work intermittently held the GIL for more than one second,
                    # delaying frame delivery and making the timeline jump.
                    traffic_rows,satellite_traffic,timings["traffic_ms"]=read_traffic_cache_frame(traffic_cache_blobs,frame_index)
                    traffic_future=None
                else:
                    traffic_future=executor.submit(read_traffic_cache_frame,traffic_cache_source,frame_index) if traffic_cache_source is not None else executor.submit(compute_traffic_rows,traffic_model,traffic_cells,positions,cfg.min_elevation_deg,frame_time)
                links = []
                max_isl_m = cfg.max_isl_km * 1000
                if candidates:
                    edge_array=candidate_array;coordinates=np.asarray(xyz);a_idx=edge_array[:,0];b_idx=edge_array[:,1];kinds=edge_array[:,2];point_a=coordinates[a_idx];delta=coordinates[b_idx]-point_a;distance_sq=np.einsum("ij,ij->i",delta,delta)
                    valid=np.ones(len(edge_array),dtype=bool)
                    if max_isl_m>0:valid&=distance_sq<=max_isl_m*max_isl_m
                    latitude=np.asarray([row[2] for row in positions]);valid&=(kinds==0)|((np.abs(latitude[a_idx])<=cfg.latitude_limit_deg)&(np.abs(latitude[b_idx])<=cfg.latitude_limit_deg))
                    denominator=np.maximum(distance_sq,1.0);closest_t=np.clip(-np.einsum("ij,ij->i",point_a,delta)/denominator,0.0,1.0);closest=point_a+closest_t[:,None]*delta;valid&=np.einsum("ij,ij->i",closest,closest)>EARTH_RADIUS_M**2
                    links=edge_array[valid].tolist()
                timings["isl_ms"]=round((time.perf_counter()-stage_start)*1000,1);stage_start=time.perf_counter()
                seams = []
                seam_pairs = seam_candidates
                if cfg.walker_layout == "star" and seam_candidates and len(shell_runs)==1:
                    source_ids = sorted({a for a, _b, _kind in seam_candidates})
                    target_plane = (cfg.seam_after_plane + 1) % cfg.planes
                    target_ids = range(target_plane * per_plane, (target_plane + 1) * per_plane)
                    seam_pairs = []
                    for a in source_ids:
                        xa, ya, za = xyz[a]
                        b = min(target_ids, key=lambda sid: (xyz[sid][0]-xa)**2 +
                                (xyz[sid][1]-ya)**2 + (xyz[sid][2]-za)**2)
                        seam_pairs.append((a, b, 1))
                for a, b, _kind in seam_pairs:
                    if abs(positions[a][2]) > cfg.latitude_limit_deg or abs(positions[b][2]) > cfg.latitude_limit_deg:
                        continue
                    xa, ya, za = xyz[a]; xb, yb, zb = xyz[b]
                    distance = math.sqrt((xa-xb)**2 + (ya-yb)**2 + (za-zb)**2)
                    if max_isl_m > 0 and distance > max_isl_m: continue
                    if not line_clear_of_earth(xyz[a], xyz[b]): continue
                    seams.append([a, b])
                gsls,timings["gsl_ms"]=gsl_future.result()
                if traffic_future is not None:traffic_rows,satellite_traffic,timings["traffic_ms"]=traffic_future.result()
                stage_start=time.perf_counter()
                priority_traffic=priority_traffic_by_satellite(traffic_rows,cfg.priority_task_regions) if planner else {}
                planning = planner.update(sim_time, positions, links, gsls, satellite_traffic,priority_traffic=priority_traffic) if planner else None
                timings["planning_ms"]=round((time.perf_counter()-stage_start)*1000,1)
                if planning:
                    active=planning.get("active_evaluation") or {}
                    active_plan=planning.get("execution_plan") or planning.get("active_plan") or {}
                    context=planning.get("decision_context") or {}
                    demand=float(context.get("traffic_demand_gbps",active_plan.get("total_demand_gbps",0)) or 0)
                    physical_capacity=float(active.get("actual_capacity_gbps",0) or 0)
                    route_allocated=sum(float(x.get("demand_gbps",0)) for x in active.get("routes",[]))
                    scheduled=float(active.get("scheduled_gbps",0) or 0)
                    delivered=float(planning.get("delivered_gbps",scheduled) or 0)
                    diagnostic={
                      "generation":planning.get("generation",0),"planning_event":bool(planning.get("planning_event")),
                      "scheduling_event":bool(planning.get("scheduling_event")),"action":planning.get("action",""),
                      "capacity_weight":context.get("capacity_weight"),"delay_weight":context.get("delay_weight"),
                      "delay_reference_ms":context.get("delay_reference_ms"),"demand_gbps":round(demand,6),
                      "planned_capacity_gbps":active.get("ideal_capacity_gbps",active_plan.get("total_capacity_gbps",0)),
                      "physical_capacity_gbps":round(physical_capacity,6),"route_allocated_gbps":round(route_allocated,6),
                      "scheduled_gbps":round(scheduled,6),"delivered_gbps":round(delivered,6),
                      "capacity_shortfall_gbps":round(max(0.,demand-physical_capacity),6),
                      "routing_shortfall_gbps":round(max(0.,min(demand,physical_capacity)-route_allocated),6),
                      "execution_shortfall_gbps":round(max(0.,route_allocated-scheduled),6),
                      "switching_loss_gbps":planning.get("switching_loss_gbps",0),
                      "traffic_satellite_count":context.get("traffic_satellite_count",0),
                      "routed_satellite_count":active.get("routed_source_count",active.get("route_count",0)),"route_count":active.get("route_count",0),"planned_link_count":context.get("planned_link_count",0),
                      "active_link_count":context.get("active_link_count",0),"route_available_ratio":context.get("route_available_ratio",0),
                      "business_route_satisfaction_ratio":context.get("business_route_satisfaction_ratio",0),
                      "route_capacity_gap_gbps":context.get("route_capacity_gap_gbps",0),
                      "capacity_achievement_ratio":context.get("capacity_achievement_ratio",0),
                      "unrouted_traffic_satellite_count":context.get("unrouted_traffic_satellite_count",0),
                      "unrouted_business_gbps":context.get("unrouted_business_gbps",0),
                      # Keep frame diagnostics compact. Full source lists and
                      # the 80-link selection signature were being written on
                      # every frame and added avoidable storage latency.
                      "unrouted_satellite_ids":list(context.get("unrouted_satellite_ids",[]))[:20],
                      "unrouted_reason":context.get("unrouted_reason","无"),
                      "weighted_delay_ms":active.get("business_propagation_delay_ms",0),
                      "p95_delay_ms":active.get("p95_propagation_delay_ms",0),
                      "constraint_source":context.get("constraint_source",""),
                      "objective":active_plan.get("objective_diagnostics",{}) if planning.get("planning_event") else {}}
                    append_diagnostic(diagnostic_path,"planning_diagnostic",frame=frame_index,sim_time_s=sim_time,**diagnostic)
                compute_ms = (time.perf_counter() - wall_start) * 1000
                station_counts=[0]*len(cfg.stations);elevations=[]
                for gid,_sid,elevation in gsls:
                    if 0<=gid<len(station_counts):station_counts[gid]+=1
                    elevations.append(elevation)
                same_count=sum(1 for row in links if row[2]==0);cross_count=len(links)-same_count
                frame_summary={"index":frame_index,"time_s":sim_time,"absolute_time":frame_time.isoformat(),"compute_ms":round(compute_ms,1),"satellite_count":len(positions),"same_isl_count":same_count,"cross_isl_count":cross_count,"seam_disabled_count":len(seams),"gsl_count":len(gsls),"visible_satellite_count":len({row[1] for row in gsls}),"active_gateway_count":sum(x>0 for x in station_counts),"average_elevation_deg":round(sum(elevations)/len(elevations),4) if elevations else "","max_elevation_deg":round(max(elevations),4) if elevations else "","station_visible_counts":station_counts}
                frame_payload={"index": frame_index, "total": total_frames, "time_s": sim_time,
                    "absolute_time": (epoch + timedelta(seconds=sim_time)).isoformat(),
                    "satellites": positions, "isls": links, "seams": seams,
                    "gsls": gsls, "traffic": traffic_rows,
                    "satellite_traffic": satellite_traffic,
                    "priority_satellite_traffic": [[sid,amount] for sid,amount in priority_traffic.items() if amount>0],
                    "planning": planning,
                    "summary":frame_summary,
                    "timings_ms":timings,
                    "compute_ms": round(compute_ms, 1)}
                with self._control_lock:self._latest_full_frame=frame_payload
                with self._control_lock:display_options=dict(self._display_options)
                display_started=time.perf_counter();display_frame=MainWindow._filter_frame(frame_payload,display_options);display_filter_ms=(time.perf_counter()-display_started)*1000
                display_frame["_display_filtered"]=True;display_frame["_emitted_perf"]=time.perf_counter();display_frame["_render_filter_ms"]=round(display_filter_ms,1)
                self.display_frame_ready.emit(display_frame)
                # Result compression used to run in the writer thread.  Its Python work
                # intermittently held the GIL and delayed otherwise quick frames without
                # appearing in the displayed compute time.  Do it here as an explicit,
                # measurable frame stage; the writer only performs SQLite I/O.
                storage_started=time.perf_counter()
                result_blob=b"M"+zlib.compress(marshal.dumps(frame_payload),1) if self.result_store is not None else None
                storage_ms=(time.perf_counter()-storage_started)*1000
                timings["storage_ms"]=round(storage_ms,1)
                frame_payload["process_ms"]=round(compute_ms+storage_ms,1)
                if result_blob is not None:frame_payload["_result_blob"]=result_blob
                if self.result_store is not None:self.result_store.write_frame(frame_payload)
                interval_ms=1000*cfg.step_s/max(0.01,self._speed)
                append_diagnostic(diagnostic_path,"frame_computed",frame=frame_index,sim_time_s=sim_time,interval_budget_ms=round(interval_ms,1),process_ms=frame_payload["process_ms"],timings_ms=timings,satellite_count=len(positions),same_isl_count=same_count,cross_isl_count=cross_count,gsl_count=len(gsls),traffic_cell_count=len(traffic_rows))
                # Qt result tables do not need a 20,000-satellite full payload
                # every simulated second. Queuing this large dictionary on
                # every frame let the GUI fall progressively behind; queued
                # table updates retained full position arrays/result blobs and
                # competed for the GIL, which made even the measured orbit
                # stage rise from ~16 ms to >1 s. Refresh result pages every
                # five simulated seconds, and always on a planning event/final
                # frame. The independently filtered map stream remains live.
                result_view_due=(frame_index%max(1,int(round(5/max(cfg.step_s,1e-9))))==0 or
                                 frame_index==total_frames-1 or bool((planning or {}).get("planning_event")))
                if result_view_due:
                    result_view_frame={key:value for key,value in frame_payload.items() if key!="_result_blob"}
                    self.frame_ready.emit(result_view_frame)
                self.progress.emit(round((frame_index + 1) * 100 / total_frames),
                    f"第 {frame_index + 1}/{total_frames} 帧，总处理 {compute_ms+storage_ms:.1f} ms｜模型 {compute_ms:.1f}｜存储编码 {storage_ms:.1f}｜轨道 {timings['orbit_ms']:.1f}｜ISL {timings['isl_ms']:.1f}｜GSL {timings['gsl_ms']:.1f}｜业务 {timings['traffic_ms']:.1f}｜规划 {timings['planning_ms']:.1f} ms")
                next_frame = frame_index + 1
                if next_frame >= total_frames:
                    next_frame=total_frames-1
                    with self._control_lock:self._paused=True
                    self.paused_changed.emit(True)
                # Keep simulation time tied to one monotonic deadline.  Recomputing
                # and then undoing the deadline on every 30 ms wakeup introduced
                # small cadence errors that became visible on the timeline.
                with self._control_lock:
                    speed=max(0.01,self._speed)
                interval=cfg.step_s/speed
                if not first_frame_submitted:
                    # Frame 0 performs the initial global plan. Starting its
                    # deadline before that work consumed most of the first
                    # one-second interval and made frames 0/1 arrive together.
                    # Start the display cadence after frame 0 is submitted.
                    next_due=time.perf_counter()+interval
                    first_frame_submitted=True
                else:
                    next_due+=interval
                append_diagnostic(diagnostic_path,"frame_schedule",frame=frame_index,next_frame=next_frame,interval_ms=round(interval*1000,1),deadline_lag_ms=round((time.perf_counter()-next_due)*1000,1),paused=self._paused)
                while not self._stop.is_set():
                    with self._control_lock:
                        if self._requested_frame is not None or self._paused: break
                    now=time.perf_counter();remaining=next_due-now
                    if remaining < -interval:
                        next_due=now;remaining=0
                    if remaining <= 0: break
                    self._stop.wait(min(0.03, remaining))
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        finally:
            if "gc_was_enabled" in locals() and gc_was_enabled and not gc.isenabled():
                gc.enable();gc.collect()
            if "executor" in locals():executor.shutdown(wait=False,cancel_futures=True)
            if "traffic_cache_connection" in locals() and traffic_cache_connection is not None:traffic_cache_connection.close()
            self.finished.emit()

    def stop(self): self._stop.set()
    def seek(self, frame_index):
        with self._control_lock: self._requested_frame = frame_index
    def set_paused(self, paused):
        with self._control_lock: self._paused = paused
    def set_speed(self, speed):
        with self._control_lock: self._speed = speed


class MainWindow(QMainWindow):
    filtered_frame_ready = Signal(dict, int)

    def __init__(self, *, load_map=True):
        super().__init__(); self.setWindowTitle("星地馈电链路与星座仿真验证系统")
        area = self.screen().availableGeometry()
        self.resize(min(1500,max(1,area.width()-32)), min(900,max(1,area.height()-80)))
        icon_path=Path(__file__).with_name("assets")/"satellite.svg"
        if icon_path.exists(): self.setWindowIcon(QIcon(str(icon_path)))
        self.thread = self.worker = None; self.web_ready = False; self.pending_frame = None; self.transfer_queue = [];self.transfer_in_flight=False
        self.last_transfer_page = None; self.last_sent_isls = {}
        self.latest_full_frame = None; self.current_full_frame = None; self.scenario_planes = None
        self.frame_cache = OrderedDict(); self.frame_cache_limit = 120; self.follow_live = True; self.is_paused = False
        self.summary_history = {}; self.active_config = None; self.prepared_metadata = None
        self.result_store=None; self.result_store_path=None; self.requested_result_path=None; self.loaded_result=False; self.loaded_frame_indices=[]; self.result_frame_position=0; self.result_playing=False
        self.user_stopping = False; self.prepared_ready = False; self.regenerate_pending=False;self._one_click_running=False;self._run_all_after_stop=False;self._closing=False
        self.map_view_bounds=None;self.input_view_mode="topology";self.last_display_at=0.0;self.timeline_dragging=False;self.seeking_index=None;self.timeline_resume_after_drag=False
        self.render_filter_executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix="render-filter");self.render_filter_future=None;self.render_filter_generation=0
        self.filtered_frame_ready.connect(self._filtered_frame_ready)
        self.run_started_at=None;self.run_elapsed_s=0.0;self.preparation_started_at=None;self.offline_planning_started_at=None;self.elapsed_phase="idle";self._simulation_clock_pending=False; self._auto_left_collapsed=False; self._auto_right_collapsed=False; self._panel_animations=[]; self._last_cpu_time=time.process_time(); self._last_cpu_wall=time.perf_counter()
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground,True)
        self._build_ui(); self._apply_dashboard_style()
        self._enable_dark_title_bar()
        self.web.settings().setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
        self.web.settings().setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, False)
        self.notice_banner=QLabel(self);self.notice_banner.setObjectName("noticeBanner");self.notice_banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.notice_banner.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents,True);self.notice_banner.hide()
        self.web_channel=QWebChannel(self.web.page()); self.map_bridge=MapBridge(self); self.map_bridge.plane_filter_changed.connect(self._map_plane_filter_changed);self.map_bridge.viewport_changed.connect(self._map_viewport_changed);self.map_bridge.traffic_details_requested.connect(self._traffic_details_requested);self.map_bridge.satellite_path_requested.connect(self._satellite_path_requested);self.map_bridge.satellite_exit_comparison_requested.connect(self._satellite_exit_comparison_requested);self.map_bridge.input_view_changed.connect(self._input_view_changed); self.web_channel.registerObject("mapBridge",self.map_bridge); self.web.page().setWebChannel(self.web_channel)
        self.web.loadFinished.connect(self._web_loaded)
        if load_map:self.load_map()
        self.resource_timer=QTimer(self); self.resource_timer.timeout.connect(self._update_runtime_status); self.resource_timer.start(1000)
        self.result_play_timer=QTimer(self); self.result_play_timer.timeout.connect(self._advance_result_playback)

    def load_map(self):
        self.web.setUrl(QUrl.fromLocalFile(str(WEB_FILE)))

    def _enable_dark_title_bar(self):
        if sys.platform != "win32":return
        try:
            import ctypes
            enabled=ctypes.c_int(1)
            hwnd=int(self.winId()); dwm=ctypes.windll.dwmapi
            result=dwm.DwmSetWindowAttribute(hwnd,20,ctypes.byref(enabled),ctypes.sizeof(enabled))
            if result != 0:dwm.DwmSetWindowAttribute(hwnd,19,ctypes.byref(enabled),ctypes.sizeof(enabled))
            # Windows 11/新版 Windows 10：显式指定标题栏、标题文字和边框颜色。
            for attribute,color in ((35,0x001F1307),(36,0x00F7ECD9),(34,0x00664D1A)):
                value=ctypes.c_uint(color); dwm.DwmSetWindowAttribute(hwnd,attribute,ctypes.byref(value),ctypes.sizeof(value))
        except Exception:
            pass

    def showEvent(self,event):
        self._enable_dark_title_bar()
        super().showEvent(event)
        QTimer.singleShot(0,self._enforce_splitter_layout)

    def set_fullscreen(self, enabled):
        if enabled:
            self.showFullScreen()
        else:
            self.showMaximized()

    def exit_fullscreen(self):
        if self.isFullScreen():
            self.set_fullscreen(False)

    def changeEvent(self,event):
        super().changeEvent(event)
        if event.type() == QEvent.Type.WindowStateChange and hasattr(self,"fullscreen_action"):
            self.fullscreen_action.setChecked(self.isFullScreen())
            self.exit_fullscreen_action.setEnabled(self.isFullScreen())

    def _apply_dashboard_style(self):
        self.setStyleSheet("""
            QMainWindow,QWidget{background:#07131f;color:#d9ecf7;font-family:'Microsoft YaHei';font-size:14px}
            QMenuBar,QMenu,QToolBar,QStatusBar{background:#0a1b2a;color:#d9ecf7;border-color:#18384d}
            QToolBar{spacing:5px;padding:4px;border:0}
            QToolButton{background:transparent;color:#cce8f5;border:1px solid transparent;border-radius:3px;padding:5px}
            QToolButton:hover{background:#123d54;border-color:#1d6f8d} QToolButton:pressed{background:#0c2c3d}
            QWidget#projectRail{background:#081b29;border-right:1px solid #1a526d}
            QWidget#propertyRail{background:#081b29;border-left:1px solid #1a526d}
            QToolButton#verticalProjectTab,QToolButton#verticalPropertyTab{background:#0b2636;color:#58d9ed;border:0;border-bottom:1px solid #163d50;border-radius:0;padding:8px 10px;font-size:14px;font-weight:700;text-align:left}
            QToolButton#verticalProjectTab:hover,QToolButton#verticalPropertyTab:hover{background:#10384b;color:#ffffff}
            QToolButton#verticalProjectTab:checked{background:#0e3b4e;color:#7eecf8;border-left:4px solid #2ddff5}
            QWidget#secondaryProjectNav{background:#071923;border:0;border-left:1px solid #205066;border-radius:0}
            QToolButton#secondaryProjectTab{background:transparent;color:#91a9b8;border:0;border-radius:3px;padding:7px 8px 7px 14px;text-align:left;font-size:13px;font-weight:400}
            QToolButton#secondaryProjectTab:hover{background:#0c3041;color:#dff8ff}
            QToolButton#secondaryProjectTab:checked{background:#0d5268;color:#ffffff;border-left:3px solid #42e8ff;font-weight:600}
            QToolButton#verticalPropertyTab:checked{background:#105b73;color:#ffffff;border-left:3px solid #42e8ff}
            QTabWidget::pane{border:1px solid #1d4a63;background:#081724}
            QTabBar::tab{background:#0d2638;color:#91afc2;padding:7px 14px;border:1px solid #193f55}
            QTabBar::tab:selected{color:#62e8ff;background:#11354a;border-bottom:2px solid #25d9ff}
            QWidget#topologyCommandBar{background:#07131f;border:0}
            QWidget#diagnosticControlGroup{background:#0a2434;border:1px solid #143b4e;border-radius:5px}
            QLabel#diagnosticControlLabel{color:#648a9c;padding:0 6px 0 0;font-size:11px}
            QPushButton#mapModeButton{min-width:64px;min-height:27px;padding:0 10px;margin:0;background:transparent;color:#8fb2c2;border:0;border-radius:4px}
            QPushButton#mapModeButton:hover{background:#103c50;color:#dffaff;border-color:#28aeca}
            QPushButton#mapModeButton:checked{background:#0d6078;color:#ffffff;border:1px solid #50e6ff;font-weight:600}
            QWidget#centeredTabNavigation{background:#081824;border:0}
            QStackedWidget#workspaceContextNav,QWidget#contextNavPage{background:transparent;border:0}
            QWidget#segmentedTabGroup,QWidget#planningSegmentGroup{background:#0a2434;border:1px solid #143b4e;border-radius:6px}
            QWidget#mainSegmentGroup{background:transparent;border:0}
            QWidget#mainSegmentGroup QPushButton#segmentedTabButton{min-width:100px;min-height:30px;margin:0;padding:0 12px;background:transparent;color:#789eaf;border:0;border-bottom:2px solid transparent;border-radius:0}
            QWidget#planningSegmentGroup QPushButton#segmentedTabButton{min-width:82px;min-height:29px;margin:0;padding:0 9px;background:#0a2434;color:#8baebe;border:0;border-radius:5px}
            QPushButton#segmentedTabButton:hover{background:#103c50;color:#dffaff;border-color:#28aeca}
            QWidget#mainSegmentGroup QPushButton#segmentedTabButton:checked{background:#0d2d3e;color:#62e8ff;border:0;border-bottom:2px solid #2de2ff;font-weight:600}
            QWidget#planningSegmentGroup QPushButton#segmentedTabButton:checked{background:#0d6078;color:#ffffff;border:1px solid #42dff8;font-weight:600}
            QWebEngineView{border:0;background:#07131f}
            QTreeWidget,QTableWidget,QLineEdit,QComboBox,QSpinBox,QDoubleSpinBox{background:#091b29;color:#d9ecf7;border:1px solid #1c4961;selection-background-color:#12617b}
            QLineEdit:disabled,QComboBox:disabled,QSpinBox:disabled,QDoubleSpinBox:disabled,QTableWidget:disabled{background:#0a131b;color:#4f6876;border-color:#20333e}
            QCheckBox{spacing:8px;color:#d9ecf7;padding:3px} QCheckBox:disabled{color:#9ab2bd}
            QCheckBox::indicator{width:17px;height:17px;border:2px solid #39cce8;border-radius:3px;background:#07131f}
            QCheckBox::indicator:hover{border-color:#ffffff;background:#0d3042}
            QCheckBox::indicator:unchecked:disabled{background:#0b151d;border:2px solid #526b78}
            QCheckBox::indicator:checked{background:#00d8f5;border:3px solid #d9fbff}
            QCheckBox::indicator:checked:disabled{background:#00bcd8;border:3px solid #c8f8ff}
            QPushButton{background:#0d4058;color:#dff8ff;border:1px solid #1aa9d0;border-radius:3px;padding:5px 10px}
            QPushButton:hover{background:#12617b;border-color:#5ce8ff} QPushButton:disabled{color:#506876;border-color:#29404e;background:#10202a}
            QProgressBar{background:#08141e;border:1px solid #1a455c;border-radius:3px;text-align:center;color:#d9ecf7}
            QProgressBar::chunk{background:#11bddd}
            QSplitter::handle{background:transparent;width:0px}
            QLabel#propertyTitle{font-size:14px;font-weight:600;color:#5cecff;background:#0b2638;border-left:3px solid #21d9ff;padding:8px}
            QLabel#statusSegment{background:#0b2638;border:1px solid #1a4d66;border-radius:3px;padding:3px 8px;color:#a9d7e8}
            QLabel#planningKpi{background:#0a2839;border-left:3px solid #27dfff;border-radius:3px;padding:8px 10px;color:#d8f5ff}
            QLabel#noticeBanner{background:#102a3a;color:#d9edf6;border:1px solid #28617a;border-left:4px solid #45b8d3;border-radius:5px;padding:10px 18px;font-size:12px;font-weight:600}
            QMenuBar::item:selected,QMenu::item:selected{background:#12526c;color:#ffffff}
        """)

    def _spin(self, minimum, maximum, value, decimals=None, suffix=""):
        from link_sim.ui.numeric_inputs import hide_step_buttons
        box = QDoubleSpinBox() if decimals is not None else QSpinBox()
        hide_step_buttons(box)
        box.setRange(minimum, maximum); box.setValue(value)
        if decimals is not None: box.setDecimals(decimals)
        box.setSuffix(suffix); return box

    def _action(self,text,standard_icon,callback):
        action=QAction(self.style().standardIcon(standard_icon),text,self); action.triggered.connect(callback); return action

    def _notify_pending(self,name):
        QMessageBox.information(self,name,f"{name}功能已预留，相关数据模型接入后启用。")

    def _show_notice(self,text,duration_ms=6500):
        self.notice_banner.setText(text);self.notice_banner.adjustSize()
        width=min(max(520,self.notice_banner.width()),max(520,self.width()-80));self.notice_banner.resize(width,self.notice_banner.height())
        self.notice_banner.move(max(20,(self.width()-width)//2),72);self.notice_banner.raise_();self.notice_banner.show()
        QTimer.singleShot(duration_ms,self.notice_banner.hide)

    def _export_scene_config(self):
        try: data=asdict(self._config())
        except Exception as exc: QMessageBox.warning(self,"场景参数错误",str(exc)); return
        if data["source"] == "walker":
            layout_name="WalkerStar" if data["walker_layout"]=="star" else "WalkerDelta"
            source_name=f"多壳层-{layout_name}" if data.get("walker_mode") in ("multi_shell","document_4_shell") else layout_name
        else:
            source_name=f'TLE-{Path(data["tle_path"]).stem or "未命名"}'
        isl_name="3ISL砖墙" if data["isl_pattern"]=="3isl_brick" else "4ISL网格"
        stamp=datetime.now().strftime("%Y%m%d_%H%M%S")
        filename=(f'场景_{source_name}_{data["satellite_count"]}星_{data["planes"]}轨_'
                  f'{data["altitude_km"]:g}km_倾角{data["inclination_deg"]:g}度_{isl_name}_'
                  f'纬度门限{data["latitude_limit_deg"]:g}度_时长{data["duration_s"]:g}s_'
                  f'步进{data["step_s"]:g}s_{len(data["stations"])}站_{stamp}.json')
        for char in '<>:"/\\|?*':filename=filename.replace(char,"-")
        path,_=QFileDialog.getSaveFileName(self,"导出场景",str(ROOT/filename),"JSON (*.json)")
        if path:
            Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8"); self.status.setText(f"场景已导出：{path}")
        self._refresh_cesium_view()

    def _import_scene_config(self):
        if self.worker is not None:
            QMessageBox.information(self,"仿真正在运行","请先结束当前仿真，再导入场景。"); return
        path,_=QFileDialog.getOpenFileName(self,"导入场景",str(ROOT),"场景文件 (*.json);;所有文件 (*)")
        if not path:self._refresh_cesium_view(); return
        try:
            data=json.loads(Path(path).read_text(encoding="utf-8"))
            self.source.setCurrentIndex(0 if data.get("source","walker")=="walker" else 1)
            self.tle_path.setText(str(data.get("tle_path","")))
            walker_mode=str(data.get("walker_mode","single"));self.walker_mode.setCurrentIndex(0 if walker_mode in ("multi_shell","document_4_shell") else 1)
            imported_shells=data.get("walker_shells",[]);self.shell_count.setValue(max(1,len(imported_shells))) if imported_shells else None
            for row,shell in enumerate(imported_shells):
                for col,key in enumerate(("name","altitude_km","inclination_deg","planes","per_plane")):
                    self.shell_table.item(row,col).setText(str(shell.get(key,self.shell_table.item(row,col).text())))
            self.sat_count.setValue(int(data.get("satellite_count",self.sat_count.value())))
            self.planes.setValue(int(data.get("planes",self.planes.value())))
            self.walker_layout.setCurrentIndex(1 if data.get("walker_layout")=="star" else 0)
            self.altitude.setValue(float(data.get("altitude_km",self.altitude.value())))
            self.inclination.setValue(float(data.get("inclination_deg",self.inclination.value())))
            eccentricity=float(data.get("eccentricity",0.0000001)); self.orbit_shape.setCurrentIndex(0 if eccentricity<=0.0000001 else 1); self.eccentricity.setValue(eccentricity)
            self.arg_perigee.setValue(float(data.get("arg_perigee_deg",self.arg_perigee.value())))
            self.satellite_antennas.setValue(int(data.get("satellite_antenna_count",4)));self.satellite_channels.setValue(int(data.get("satellite_channel_count",4)));self.satellite_channel_capacity.setValue(float(data.get("satellite_channel_capacity_gbps",10)));self.geo_interference.setChecked(bool(data.get("geo_interference_enabled",True)))
            self.geo_satellite_count.setValue(int(data.get("geo_satellite_count",3)));self.geo_center_longitude.setValue(float(data.get("geo_center_longitude_deg",110)));self.geo_longitude_spacing.setValue(float(data.get("geo_longitude_spacing_deg",30)));self.geo_altitude.setValue(float(data.get("geo_altitude_km",35786)));self.gso_i_over_n.setValue(float(data.get("gso_i_over_n_db",-12)));self.gso_protection_angle.setValue(float(data.get("gso_protection_angle_deg",10)))
            self.duration.setValue(float(data.get("duration_s",self.duration.value()))); self.step.setValue(float(data.get("step_s",self.step.value())))
            speed=f'{float(data.get("speed",1)):g}×'; self.speed.setCurrentText(speed)
            self._lat_limit_auto="latitude_limit_deg" not in data; self._setting_lat_limit=True; self.lat_limit.setValue(float(data.get("latitude_limit_deg",self.lat_limit.value()))); self._setting_lat_limit=False; self._sync_default_latitude_limit(); self.elevation.setValue(float(data.get("min_elevation_deg",self.elevation.value())))
            # 旧场景中的 max_gsl_km 已废弃，不再参与星地可见性计算。
            self.max_isl.setValue(float(data.get("max_isl_km",self.max_isl.value())) or 2000.0)
            self.gateway_eirp.setValue(float(data.get("gateway_eirp_dbw",65)));self.gateway_gt.setValue(float(data.get("gateway_gt_dbk",30)));self.gateway_array_loss.setValue(float(data.get("gateway_array_loss_db",1)))
            self.satellite_eirp.setValue(float(data.get("satellite_eirp_dbw",26)))
            self.feeder_frequency.setValue(float(data.get("feeder_frequency_ghz",39)))
            self.feeder_rolloff.setValue(float(data.get("feeder_rolloff",.20)))
            self.gateway_planning_enabled.setChecked(False)
            self.gateway_planning_mode.setCurrentIndex(1)
            self.gateway_max_sites.setValue(int(data.get("gateway_max_sites",12)));self.gateway_redundancy.setValue(int(data.get("gateway_redundancy",2)));self.gateway_max_antennas.setValue(int(data.get("gateway_max_antennas",20)))
            self.traffic_enabled.setChecked(bool(data.get("traffic_enabled",True)))
            self.traffic_grid_path.setText(str(data.get("traffic_grid_path", DEFAULT_TRAFFIC_GRID)))
            self.traffic_dense_percentile.setValue(float(data.get("traffic_dense_percentile",95)))
            self.traffic_dense_max_satellites.setValue(max(4,int(data.get("traffic_dense_max_satellites",4))))
            self.traffic_random_sigma.setValue(float(data.get("traffic_random_sigma",0)))
            self.traffic_outside_china_percent.setValue(float(data.get("traffic_outside_china_percent",0)))
            self.traffic_total_gbps.setValue(float(data.get("traffic_total_gbps",500)));self.traffic_landing_percent.setValue(float(data.get("traffic_landing_percent",30)))
            self.traffic_cluster_radius.setValue(float(data.get("traffic_cluster_radius_km",180)))
            self.priority_region_table.setRowCount(0)
            for region in (data.get("priority_task_regions") or [{"name":"新疆重点区域","lat":42.0,"lon":85.0,"radius_km":900}]):self._append_priority_region(region.get("name","重点区域"),region.get("lat",0),region.get("lon",0),region.get("radius_km",100))
            self.rain_enabled.setChecked(bool(data.get("rain_enabled",True)))
            rain_sources=("geographic_auto","manual_sequence","weather_file");rain_source=str(data.get("rain_source","geographic_auto"));self.rain_source.setCurrentIndex(rain_sources.index(rain_source) if rain_source in rain_sources else 0)
            rain_sigmas=(0.50,0.35,0.20,0.12);rain_sigma=float(data.get("rain_sense_sigma",0.20));self.rain_sense.setCurrentIndex(min(range(len(rain_sigmas)),key=lambda i:abs(rain_sigmas[i]-rain_sigma)))
            self.rain_forecast.setValue(float(data.get("rain_forecast_hours",2)));self.rain_p90.setChecked(bool(data.get("rain_use_p90",True)))
            self.planning_enabled.setChecked(bool(data.get("planning_enabled",True)))
            self.planning_period.setValue(float(data.get("planning_period_s",300)))
            self.planning_horizon.setValue(float(data.get("planning_horizon_s",1200)))
            self.planning_slot.setValue(float(data.get("planning_slot_s",15)))
            self.planning_capacity_reserve.setValue(float(data.get("planning_capacity_reserve_percent",20)))
            strategy_index=self.planning_strategy.findData(str(data.get("planning_selection_strategy","legacy")))
            self.planning_strategy.setCurrentIndex(max(0,strategy_index))
            self.planning_capacity_weight.setValue(100*float(data.get("planning_capacity_weight",.5)))
            self.fast_evaluation_period.setValue(float(data.get("fast_evaluation_period_s",1)))
            self.fast_adjust_period.setValue(1.0)
            self.route_update_period.setValue(float(data.get("route_update_period_s",15)))
            
            self.feeder_bandwidth.setValue(float(data.get("feeder_bandwidth_mhz",500)))
            self.snr_margin.setValue(float(data.get("snr_margin_db",1)))
            
            # 切换过程不再用人为百分比扣减业务量；链路中断和重路由结果
            # 已由实际可用容量与路径状态直接反映。
            self.switching_loss.setValue(0)
            self.modcod_table_path.setText(str(data.get("modcod_table_path","")))
            self.isl_pattern.setCurrentIndex(1 if data.get("isl_pattern")=="3isl_brick" else 0)
            layer_values=((self.show_sat,"show_satellites"),(self.show_same,"show_same_isl"),(self.show_cross,"show_cross_isl"),(self.show_gsl,"show_gsl"),(self.show_motion,"show_motion"),(self.show_labels,"show_labels"))
            for widget,key in layer_values:widget.setChecked(bool(data.get(key,True)))
            stations=data.get("stations",[])
            self.station_table.blockSignals(True); self.station_table.setRowCount(0)
            for station in stations:self._append_station(str(station.get("name","地面站")),float(station.get("lat",0)),float(station.get("lon",0)),float(station.get("alt",0)),int(station.get("antenna_count",20)),int(station.get("channel_count",station.get("antenna_count",20))))
            self.station_table.blockSignals(False); self._refresh_scene_preview(); self.current_scene_node.setText(0,f"当前场景（{Path(path).stem}）"); self._show_property_section(0)
            self.status.setText(f"场景已导入：{path}")
        except Exception as exc:
            self.station_table.blockSignals(False); QMessageBox.critical(self,"导入场景失败",f"{type(exc).__name__}: {exc}")
        self._refresh_cesium_view()

    def _refresh_cesium_view(self):
        if hasattr(self,"web") and self.web_ready:
            self.web.page().runJavaScript("if(window.viewer){viewer.forceResize();viewer.scene.requestRender();}")

    @staticmethod
    def _result_scene_folder_name(cfg):
        source=("WalkerStar" if cfg.walker_layout=="star" else "WalkerDelta") if cfg.source=="walker" else f"TLE-{Path(cfg.tle_path).stem or '场景'}"
        name=f'场景_{source}_{cfg.satellite_count}星_{cfg.planes}轨_{cfg.duration_s:g}s_{cfg.step_s:g}s_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        for char in '<>:"/\\|?*':name=name.replace(char,"-")
        return name

    @staticmethod
    def _public_scenario(cfg):
        config=asdict(cfg);config.pop("result_directory",None);config.pop("scenario_directory",None);config.pop("speed",None)
        encoded=json.dumps(config,sort_keys=True,ensure_ascii=False,separators=(",",":"));scenario_hash=hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        source=("WalkerStar" if cfg.walker_layout=="star" else "WalkerDelta") if cfg.source=="walker" else f"TLE-{Path(cfg.tle_path).stem or '场景'}"
        shell_label=f"{len(cfg.walker_shells)}壳层" if cfg.source=="walker" and cfg.walker_shells else f"{cfg.altitude_km:g}km_{cfg.inclination_deg:g}度"
        isl_label="4ISL" if cfg.isl_pattern=="4isl_grid" else "3ISL"
        traffic_label="中国业务" if cfg.traffic_enabled and cfg.traffic_outside_china_percent<=0 else ("全球业务" if cfg.traffic_enabled else "无业务")
        gateway_label="信关站自动规划" if cfg.gateway_planning_mode=="automatic" else f"{len(cfg.stations)}站"
        label=f"{source}_{shell_label}_{cfg.satellite_count}星_{cfg.planes}轨_{isl_label}_仰角{cfg.min_elevation_deg:g}度_{gateway_label}_{traffic_label}_时长{cfg.duration_s:g}s_步长{cfg.step_s:g}s"
        for char in '<>:"/\\|?*':label=label.replace(char,"-")
        scenario_id=f"SCN-{scenario_hash[:12].upper()}";scenario_dir=ROOT/"simulation_scenarios"/f"{label}_{scenario_id}";scenario_dir.mkdir(parents=True,exist_ok=True)
        public_config={"scenario_id":scenario_id,"config_hash":scenario_hash,"name":label,"config":config}
        (scenario_dir/"scenario_config.json").write_text(json.dumps(public_config,ensure_ascii=False,indent=2),encoding="utf-8")
        return scenario_id,scenario_hash,scenario_dir

    def _open_result_package(self):
        if self.worker is not None:QMessageBox.information(self,"仿真正在运行","请先结束当前仿真再打开结果包。"); return
        self.result_play_timer.stop(); self.result_playing=False
        path=QFileDialog.getExistingDirectory(self,"选择仿真场景结果目录",str(ROOT/"simulation_results"))
        if not path:return
        try:
            if self.result_store:self.result_store.close()
            store=ResultStore(path); config_data=store.get_metadata("config"); prepared=store.get_metadata("prepared")
            if not config_data or not prepared:raise ValueError("结果包缺少场景元数据")
            self.result_store=store; self.result_store_path=Path(path); self.loaded_result=True; self.follow_live=True; self.frame_cache.clear(); self.active_config=SimulationConfig(**config_data); self.scenario_planes=prepared["planes"]
            self.prepared_metadata=dict(prepared)
            self.planning_results.set_gateway_deployment(prepared.get("gateway_deployment",{}))
            self.loaded_frame_indices=store.frame_indices()
            if not self.loaded_frame_indices:raise ValueError("结果包中尚无已保存的动态帧")
            self.display_planes.setCurrentIndex(2); self.center_plane.setValue(1)
            self.timeline.setRange(self.loaded_frame_indices[0],self.loaded_frame_indices[-1]); self.timeline.setEnabled(True); self.prepared_ready=True; self.is_paused=True
            prepared["layers"]=self._layer_options(); prepared["center_plane"]=0; prepared["display_plane_count"]=0; prepared["priority_regions"]=list(self.active_config.priority_task_regions); self.web.page().runJavaScript(f"configureScenario({json.dumps(prepared,ensure_ascii=False)})")
            self.result_frame_position=0; self.result_playing=False; self.result_play_timer.stop()
            self.prepare_btn.setEnabled(False);self.start_btn.setText("播放结果"); self.start_btn.setEnabled(True); self.stop_btn.setEnabled(True); self.state_badge.setText("● 正在读取"); self.status.setText(f"正在读取结果包首帧：{path}")
            self._sync_run_controls()
            QTimer.singleShot(0,lambda:self._load_result_frame(self.loaded_frame_indices[0]))
        except Exception as exc:
            QMessageBox.critical(self,"打开结果包失败",f"{type(exc).__name__}: {exc}")

    def _load_result_frame(self,index):
        if not self.loaded_result or not self.result_store:return
        actual,frame=self.result_store.nearest_frame(index)
        if frame is None:self.status.setText("结果包中没有可读取的动态帧"); return
        self._queue_frame(frame); self.timeline.setValue(actual)
        try:self.result_frame_position=self.loaded_frame_indices.index(actual)
        except ValueError:self.result_frame_position=0
        self.state_badge.setText("● 回放中" if self.result_playing else "● 已暂停")
        suffix="" if actual==index else f"（请求帧{index}未保存，已定位到最近的第{actual}帧）"
        self.status.setText(f"已读取 t={frame['time_s']:g} s {suffix}")

    def _result_interval_ms(self):
        step=self.active_config.step_s if self.active_config else 1.0; speed=float(self.speed.currentText().replace("×","")); return max(30,round(step*1000/speed))

    def _toggle_result_playback(self):
        if not self.loaded_frame_indices:return
        self.result_playing=not self.result_playing
        if self.result_playing:
            self.result_play_timer.start(self._result_interval_ms()); self.start_btn.setText("暂停结果"); self.state_badge.setText("● 回放中"); self.status.setText("正在回放已保存的仿真结果")
        else:
            self.result_play_timer.stop(); self.start_btn.setText("播放结果"); self.state_badge.setText("● 已暂停"); self.status.setText("结果回放已暂停，可拖动时间轴")

    def _advance_result_playback(self):
        if not self.loaded_result or not self.loaded_frame_indices:return
        self.result_frame_position=(self.result_frame_position+1)%len(self.loaded_frame_indices); self._load_result_frame(self.loaded_frame_indices[self.result_frame_position])

    def _pause_simulation(self):
        if self.loaded_result and self.result_playing:self._toggle_result_playback(); return
        if self.worker is not None and not self.is_paused:self._toggle_pause()

    def _reset_layout(self):
        self._set_project_drawer(False); self._set_property_drawer(False)
        self.main_splitter.setSizes([52,max(600,self.width()-52),0]); self.status.setText("已恢复默认界面布局")

    def _save_layout(self):
        settings=QSettings("RXZL","SatelliteFeederSimulator"); settings.setValue("windowGeometry",self.saveGeometry()); settings.setValue("windowState",self.saveState()); settings.setValue("splitterSizes",self.main_splitter.sizes()); settings.setValue("propertyDrawerExpanded",self._drawer_expanded(self.property_panel)); settings.setValue("propertyPage",self._active_property_index); settings.setValue("projectDrawerExpanded",self._drawer_expanded(self.project_panel)); settings.setValue("projectPage",self._active_project_index); self.status.setText("当前布局已保存")

    def _restore_layout(self):
        settings=QSettings("RXZL","SatelliteFeederSimulator"); geometry=settings.value("windowGeometry"); state=settings.value("windowState"); sizes=settings.value("splitterSizes")
        if geometry:self.restoreGeometry(geometry)
        if state:self.restoreState(state)
        if sizes:self.main_splitter.setSizes([int(x) for x in sizes])
        self._active_project_index=int(settings.value("projectPage",1)); self._set_project_drawer(str(settings.value("projectDrawerExpanded","false")).lower() in ("true","1"))
        self._active_property_index=int(settings.value("propertyPage",0)); self._set_property_drawer(False)
        self.status.setText("已恢复保存的布局")

    def _toggle_panel(self,panel,width):
        if panel is self.property_panel:
            self._set_property_drawer(not self._drawer_expanded(panel)); return
        if panel is self.project_panel:
            self._set_project_drawer(not self._drawer_expanded(panel)); return
        action=self.left_panel_action if panel is self.project_panel else self.right_panel_action
        self._set_panel_visible(panel,not panel.isVisible() or panel.maximumWidth()==0,width,action)

    def _set_panel_visible(self,panel,visible,width,action):
        if visible:panel.show()
        start=panel.width() if panel.isVisible() else 0; end=width if visible else 0
        panel.setMinimumWidth(0); panel.setMaximumWidth(max(start,1))
        animation=QPropertyAnimation(panel,b"maximumWidth",self); animation.setDuration(220); animation.setStartValue(start); animation.setEndValue(end); animation.setEasingCurve(QEasingCurve.Type.InOutCubic)
        if not visible:animation.finished.connect(panel.hide)
        animation.finished.connect(lambda:self._panel_animations.remove(animation) if animation in self._panel_animations else None)
        self._panel_animations.append(animation); action.setChecked(visible); animation.start()

    def resizeEvent(self,event):
        super().resizeEvent(event); width=event.size().width()
        if not hasattr(self,"property_panel") or not hasattr(self,"right_panel_action"):return
        if width < 1350 and self._drawer_expanded(self.property_panel) and self.right_panel_action.isChecked():
            self._auto_right_collapsed=True; self._set_property_drawer(False)
        elif width >= 1450 and self._auto_right_collapsed:
            self._auto_right_collapsed=False; self._set_property_drawer(True)
        if width < 1000 and self._drawer_expanded(self.project_panel) and self.left_panel_action.isChecked():
            self._auto_left_collapsed=True; self._set_project_drawer(False)
        elif width >= 1100 and self._auto_left_collapsed:
            self._auto_left_collapsed=False; self._set_project_drawer(True)
        QTimer.singleShot(0,self._enforce_splitter_layout)

    def _update_runtime_status(self):
        if self.elapsed_phase=="preparation" and self.preparation_started_at is not None:
            seconds=max(0,int(time.perf_counter()-self.preparation_started_at));self.elapsed_label.setText(f"准备耗时 {seconds//3600:02d}:{seconds//60%60:02d}:{seconds%60:02d}")
        elif self.elapsed_phase=="simulation":
            elapsed=self.run_elapsed_s+(time.perf_counter()-self.run_started_at if self.run_started_at is not None else 0.0);seconds=max(0,int(elapsed));self.elapsed_label.setText(f"仿真耗时 {seconds//3600:02d}:{seconds//60%60:02d}:{seconds%60:02d}")
        try:
            import psutil
            if not hasattr(self,"_process_monitor"):
                self._process_monitor=psutil.Process(os.getpid());self._process_monitor.cpu_percent(None);self._child_process_monitors={}
            children={child.pid:child for child in self._process_monitor.children(recursive=True)}
            monitors={};child_cpu=0.0;child_memory=0
            for pid,child in children.items():
                monitor=self._child_process_monitors.get(pid)
                if monitor is None:monitor=child;monitor.cpu_percent(None)
                monitors[pid]=monitor
                try:child_cpu+=monitor.cpu_percent(None);child_memory+=monitor.memory_info().rss
                except (psutil.NoSuchProcess,psutil.AccessDenied):pass
            self._child_process_monitors=monitors
            cpu=(self._process_monitor.cpu_percent(None)+child_cpu)/max(1,os.cpu_count() or 1)
            memory=(self._process_monitor.memory_info().rss+child_memory)/1024/1024
            process_text=f" / {len(monitors)+1}进程" if monitors else ""
            self.resource_label.setText(f"资源 CPU {cpu:.1f}%{process_text} / 内存 {memory:.0f} MB")
        except Exception:
            now_cpu=time.process_time(); now_wall=time.perf_counter(); elapsed=max(1e-6,now_wall-self._last_cpu_wall); cpu=max(0,(now_cpu-self._last_cpu_time)/elapsed*100/max(1,os.cpu_count() or 1)); self._last_cpu_time=now_cpu; self._last_cpu_wall=now_wall
            memory_text="--"
            if sys.platform=="win32":
                try:
                    import ctypes
                    from ctypes import wintypes
                    class PMC(ctypes.Structure):
                        _fields_=[("cb",wintypes.DWORD),("PageFaultCount",wintypes.DWORD),("PeakWorkingSetSize",ctypes.c_size_t),("WorkingSetSize",ctypes.c_size_t),("QuotaPeakPagedPoolUsage",ctypes.c_size_t),("QuotaPagedPoolUsage",ctypes.c_size_t),("QuotaPeakNonPagedPoolUsage",ctypes.c_size_t),("QuotaNonPagedPoolUsage",ctypes.c_size_t),("PagefileUsage",ctypes.c_size_t),("PeakPagefileUsage",ctypes.c_size_t)]
                    counters=PMC(); counters.cb=ctypes.sizeof(PMC); ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(),ctypes.byref(counters),counters.cb); memory_text=f"{counters.WorkingSetSize/1024/1024:.0f} MB"
                except Exception:pass
            self.resource_label.setText(f"资源 CPU {cpu:.1f}% / 内存 {memory_text}")

    def _build_ui(self):
        splitter = QSplitter(); self.main_splitter=splitter; splitter.setChildrenCollapsible(False); splitter.setHandleWidth(0); self.setCentralWidget(splitter)
        project_panel = QWidget(); self.project_panel=project_panel; project_panel.setFixedWidth(52); project_panel.setSizePolicy(QSizePolicy.Policy.Fixed,QSizePolicy.Policy.Expanding); project_panel.setProperty("drawerExpanded",False); project_outer=QHBoxLayout(project_panel); project_outer.setContentsMargins(0,0,0,0); project_outer.setSpacing(2)
        self.project_rail=QWidget(); self.project_rail.setObjectName("projectRail"); self.project_rail.setFixedWidth(48); self.project_rail_layout=QVBoxLayout(self.project_rail); self.project_rail_layout.setContentsMargins(3,4,3,4); self.project_rail_layout.setSpacing(4); project_outer.addWidget(self.project_rail)
        self.project_drawer=QWidget(); self.project_drawer.hide(); project_layout=QVBoxLayout(self.project_drawer); project_layout.setContentsMargins(4,4,4,4); project_outer.addWidget(self.project_drawer,1)
        self.project_tree = QTreeWidget(); self.project_tree.setHeaderHidden(True); project_layout.addWidget(self.project_tree)
        project = QTreeWidgetItem(self.project_tree,["仿真场景"]); self.project_root=project;self.link_layer_tree=project;self.network_layer_tree=project
        network_models=QTreeWidgetItem(project,["基础模型"])
        constellation=QTreeWidgetItem(network_models,["星座建模"]); constellation.setData(0,Qt.ItemDataRole.UserRole,0)
        topology_model=QTreeWidgetItem(network_models,["星间链路建模"]); topology_model.setData(0,Qt.ItemDataRole.UserRole,1)
        gateway_model=QTreeWidgetItem(network_models,["信关站建模"]); gateway_model.setData(0,Qt.ItemDataRole.UserRole,2)
        traffic_model_item=QTreeWidgetItem(network_models,["业务建模"]); traffic_model_item.setData(0,Qt.ItemDataRole.UserRole,3)
        rain_model_item=QTreeWidgetItem(network_models,["雨衰建模"]); rain_model_item.setData(0,Qt.ItemDataRole.UserRole,4)
        scene=QTreeWidgetItem(project,["当前场景（未生成）"]);self.network_scenes_tree=scene;self.current_scene_node=scene;scene.setData(0,Qt.ItemDataRole.UserRole,0)
        scene_details=QTreeWidgetItem(scene,["星座模型"]); self.scene_tree=scene_details; scene_details.setData(0,Qt.ItemDataRole.UserRole,0)
        self.isl_tree=QTreeWidgetItem(scene,["星间链路模型"]); self.isl_tree.setData(0,Qt.ItemDataRole.UserRole,1)
        gateways=QTreeWidgetItem(scene,["信关站模型"]); self.gateway_tree=gateways; gateways.setData(0,Qt.ItemDataRole.UserRole,2)
        self.traffic_tree=QTreeWidgetItem(scene,["业务模型"]); self.traffic_tree.setData(0,Qt.ItemDataRole.UserRole,3)
        self.rain_tree=QTreeWidgetItem(scene,["雨衰模型"]); self.rain_tree.setData(0,Qt.ItemDataRole.UserRole,4)
        network_tasks=QTreeWidgetItem(project,["推演任务"])
        deployment_task=QTreeWidgetItem(network_tasks,["信关站资源配置"]); deployment_task.setData(0,Qt.ItemDataRole.UserRole,2)
        planning_task=QTreeWidgetItem(network_tasks,["站星连接规划"]); planning_task.setData(0,Qt.ItemDataRole.UserRole,5)
        scheduling_task=QTreeWidgetItem(network_tasks,["实时调度执行"]); scheduling_task.setData(0,Qt.ItemDataRole.UserRole,6)
        dynamic_view=QTreeWidgetItem(network_tasks,["动态拓扑与可见性"]); dynamic_view.setData(0,Qt.ItemDataRole.UserRole,1)
        results=QTreeWidgetItem(project,["推演结果"]); self.results_tree=results
        QTreeWidgetItem(results,["慢环规划到达与连接骨架"]);QTreeWidgetItem(results,["快环调度与切换记录"]);QTreeWidgetItem(results,["容量 / 时延联合曲线"]);QTreeWidgetItem(results,["逐跳路由与天线占用"])
        project.setExpanded(True);scene.setExpanded(True)
        self.scene_info=QLabel("尚未生成基础数据"); self.scene_info.setWordWrap(True); self.scene_info.setObjectName("sceneInfo"); self.scene_info.setStyleSheet("QLabel#sceneInfo{background:#0a2131;border:1px solid #1a617c;border-radius:5px;padding:8px;color:#a9d7e8}")
        project_layout.addWidget(self.scene_info)
        self.project_buttons=[];self._project_titles={"calculation":"链路计算","simulation":"链路仿真","constellation":"星座仿真"};self._project_targets=[project]
        self._secondary_nav_containers={};self._secondary_nav_layouts={}
        for key,title in (("calculation","链路计算"),("simulation","链路仿真"),("constellation","星座仿真")):
            button=QToolButton();button.setObjectName("verticalProjectTab");button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly);button.setText("\n".join(title));button.setCheckable(True);button.setFixedWidth(42);button.setMinimumHeight(96);button.clicked.connect(lambda checked,k=key:self._primary_workspace_clicked(k));self.project_rail_layout.addWidget(button);self.project_buttons.append(button)
            secondary=QWidget();secondary.setObjectName("secondaryProjectNav");secondary_layout=QVBoxLayout(secondary);secondary_layout.setContentsMargins(16,3,4,6);secondary_layout.setSpacing(2);secondary.hide();self.project_rail_layout.addWidget(secondary)
            self._secondary_nav_containers[key]=secondary;self._secondary_nav_layouts[key]=secondary_layout
        self.project_rail_layout.addStretch();self._active_project_index=0
        panel = QWidget(); self.property_panel=panel; panel.setFixedWidth(52); panel.setSizePolicy(QSizePolicy.Policy.Fixed,QSizePolicy.Policy.Expanding); panel.setProperty("drawerExpanded",False); layout = QHBoxLayout(panel); layout.setContentsMargins(0,0,0,0); layout.setSpacing(2)
        self.property_drawer=QWidget(); self.property_drawer.hide(); drawer_layout=QVBoxLayout(self.property_drawer); drawer_layout.setContentsMargins(6,4,4,4); self.property_title=QLabel("场景参数"); self.property_title.setObjectName("propertyTitle"); drawer_layout.addWidget(self.property_title); self.property_stack=QStackedWidget(); drawer_layout.addWidget(self.property_stack,1); layout.addWidget(self.property_drawer,1)
        self.property_rail=QWidget(); self.property_rail.setObjectName("propertyRail"); self.property_rail.setFixedWidth(48); self.property_rail_layout=QVBoxLayout(self.property_rail); self.property_rail_layout.setContentsMargins(3,4,3,4); self.property_rail_layout.setSpacing(4); layout.addWidget(self.property_rail)
        orbit = QWidget(); form = QFormLayout(orbit)
        workflow_note=QLabel("先检查星座、星间链路、信关站、业务和雨衰参数，再生成基础数据。离线规划只使用本次准备的基础数据；实时调度只执行本次生成的离线方案。")
        workflow_note.setWordWrap(True);workflow_note.setObjectName("sceneInfo");form.addRow(workflow_note)
        self.source = QComboBox(); self.source.addItems(["理想Walker星座（方案设计）", "真实星座（TLE）"]); form.addRow("星座来源", self.source)
        path_row = QHBoxLayout(); self.tle_path = QLineEdit(); self.browse_tle_btn = QPushButton("选择"); self.browse_tle_btn.clicked.connect(self._browse_tle); path_row.addWidget(self.tle_path); path_row.addWidget(self.browse_tle_btn); form.addRow("TLE文件", path_row)
        self.walker_mode=QComboBox();self.walker_mode.addItems(["多壳层自定义","单壳层自定义"]);form.addRow("Walker方案",self.walker_mode)
        self.shell_count=self._spin(1,24,4);form.addRow("壳层数量",self.shell_count)
        self.shell_table=QTableWidget(4,5);self.shell_table.setHorizontalHeaderLabels(["壳层","高度 km","倾角 °","轨道面","每轨卫星"])
        for row,values in enumerate((("L1",508,53,80,100),("L2",508,70,60,100),("L3",508,90,40,100),("L4",508,30,20,100))):
            for col,value in enumerate(values):
                item=QTableWidgetItem(str(value));item.setFlags(item.flags() if col else item.flags() & ~Qt.ItemFlag.ItemIsEditable);self.shell_table.setItem(row,col,item)
        self.shell_table.setMaximumHeight(150);form.addRow("壳层参数",self.shell_table)
        self.sat_count=self._spin(1,20000,2000); self.planes=self._spin(1,500,20); self.altitude=self._spin(100,100000,508,1," km"); self.inclination=self._spin(0,180,53,2,"°")
        self.orbit_shape=QComboBox(); self.orbit_shape.addItems(["圆轨道（e=0.0000001）", "椭圆轨道"])
        self.eccentricity=self._spin(0,0.99,0.0000001,7); self.eccentricity.setEnabled(False)
        self.orbit_shape.currentIndexChanged.connect(self._orbit_shape_changed)
        self.arg_perigee=self._spin(0,360,0,2,"°")
        self.walker_layout=QComboBox(); self.walker_layout.addItems(["Walker Delta（RAAN 360°）", "Walker Star（RAAN 180°，自动启用轨道缝隙）"])
        self.duration=self._spin(1800,1800,1800,0," s"); self.duration.setToolTip("当前离线规划固定覆盖30分钟，共6个五分钟周期。")
        self.step=self._spin(0.1,3600,1,2," s"); self.step.setToolTip("仿真输出的时间步长；步长越小，帧数和计算耗时越多。")
        self.speed=QComboBox(); self.speed.addItems(["1×","2×","5×"]); self.speed.setCurrentText("1×"); self.speed.setFixedWidth(72);self.speed.setToolTip("结果播放速度，不改变轨道、规划或链路计算结果。")
        self.satellite_antennas=self._spin(1,32,4);self.satellite_channels=self._spin(1,128,4);self.satellite_channel_capacity=self._spin(0.1,1000,10,1," Gb/s");self.geo_interference=QCheckBox("纳入同频GEO/GSO干扰");self.geo_interference.setChecked(True)
        self.geo_satellite_count=self._spin(1,12,3);self.geo_center_longitude=self._spin(-180,180,110,1,"°E");self.geo_longitude_spacing=self._spin(1,60,30,1,"°");self.geo_altitude=self._spin(35000,37000,35786,0," km");self.gso_i_over_n=self._spin(-30,0,-12,1," dB");self.gso_protection_angle=self._spin(0,30,10,1,"°")
        self.isl_pattern=QComboBox(); self.isl_pattern.addItems(["4-ISL规则网格", "3-ISL固定砖墙"])
        self.seam_after_plane=self._spin(1,self.planes.value(),self.planes.value()); self.seam_after_plane.setEnabled(False)
        self.seam_after_plane.setToolTip("缝隙位于该轨道面与下一个轨道面之间；最后一面之后为第1面")
        self.planes.valueChanged.connect(self.seam_after_plane.setMaximum)
        for label,widget in [("卫星总数",self.sat_count),("轨道面数",self.planes),("Walker构型",self.walker_layout),("轨道高度",self.altitude),("轨道倾角",self.inclination),("轨道形状",self.orbit_shape),("偏心率",self.eccentricity),("近地点幅角",self.arg_perigee),("单星天线数",self.satellite_antennas),("单星通道数",self.satellite_channels),("单通道能力",self.satellite_channel_capacity)]: form.addRow(label,widget)
        form.addRow(self.geo_interference);form.addRow("参考GEO星数",self.geo_satellite_count);form.addRow("GEO中心经度",self.geo_center_longitude);form.addRow("GEO经度间隔",self.geo_longitude_spacing);form.addRow("GEO轨道高度",self.geo_altitude);form.addRow("单星GSO I/N",self.gso_i_over_n);form.addRow("GSO保护角",self.gso_protection_angle)
        self._geo_parameter_widgets=(self.geo_satellite_count,self.geo_center_longitude,self.geo_longitude_spacing,self.geo_altitude,self.gso_i_over_n,self.gso_protection_angle)
        self.geo_interference.toggled.connect(lambda enabled:[widget.setEnabled(enabled) for widget in self._geo_parameter_widgets])
        self.walker_mode.currentIndexChanged.connect(self._walker_mode_changed);self.shell_count.valueChanged.connect(self._shell_count_changed);self.shell_table.itemChanged.connect(self._walker_shells_changed);self._walker_mode_changed(0)
        self.property_stack.addWidget(orbit)
        ground = QWidget(); gl = QVBoxLayout(ground); preset_row=QHBoxLayout(); self.station_preset=QComboBox(); self.station_preset.addItems(PRESET_STATIONS); self.add_station_btn=QPushButton("添加信关站"); self.add_station_btn.clicked.connect(self._add_preset_station); preset_row.addWidget(self.station_preset); preset_row.addWidget(self.add_station_btn); gl.addLayout(preset_row)
        self.station_table=QTableWidget(0,6); self.station_table.setHorizontalHeaderLabels(["手动设置信关站","纬度","经度","高度m","天线数","信道数"]); gl.addWidget(self.station_table)
        self.remove_station_btn=QPushButton("删除信关站"); self.remove_station_btn.clicked.connect(self._remove_station); gl.addWidget(self.remove_station_btn)
        self.gateway_planning_enabled=QCheckBox();self.gateway_planning_enabled.setChecked(False);self.gateway_planning_enabled.hide()
        self.gateway_planning_mode=QComboBox();self.gateway_planning_mode.addItems(["自动规划（无需预置站点）","使用指定信关站"])
        self.gateway_planning_mode.setCurrentIndex(1);self.gateway_planning_mode.hide()
        self.gateway_max_sites=self._spin(1,100,12);self.gateway_redundancy=self._spin(1,1,1);self.gateway_redundancy.hide();self.gateway_max_antennas=self._spin(1,256,20)
        self.gateway_max_sites.hide();self.gateway_max_antennas.hide()
        self.gateway_planning_mode.currentIndexChanged.connect(self._gateway_mode_changed)
        visibility_group=QGroupBox("站星可见性约束"); gf=QFormLayout(visibility_group)
        self.elevation=self._spin(0,90,15,1,"°"); self.elevation.setSingleStep(1.0); self.elevation.setKeyboardTracking(False); self.elevation.setToolTip("默认最低工作仰角为15°；可见性由卫星实际位置计算")
        self.gsl_distance_hint=QLabel();self.gsl_distance_hint.setWordWrap(True);self.gsl_distance_hint.setStyleSheet("color:#82b8ca;padding:3px")
        self.elevation.valueChanged.connect(self._update_gsl_distance_hint);self.altitude.valueChanged.connect(self._update_gsl_distance_hint)
        gf.addRow("最低可用仰角",self.elevation);gf.addRow("参考最大斜距",self.gsl_distance_hint); gl.addWidget(visibility_group); self.property_stack.addWidget(ground)
        self._update_gsl_distance_hint()
        station_rf_group=QGroupBox("信关站链路参数");station_rf=QFormLayout(station_rf_group)
        self.gateway_eirp=self._spin(0,100,65,1," dBW");self.gateway_gt=self._spin(-20,80,30,1," dB/K");self.gateway_array_loss=self._spin(0,20,1,1," dB")
        self.satellite_eirp=self._spin(-20,80,26,1," dBW")
        station_rf.addRow("卫星下行EIRP（参考值）",self.satellite_eirp)
        station_rf.addRow("信关站接收G/T",self.gateway_gt)
        station_rf.addRow("接收系统损失",self.gateway_array_loss)
        station_rf.addRow("信关站上行EIRP（未参与下行预算）",self.gateway_eirp)
        gl.insertWidget(2,station_rf_group)
        traffic_page=QWidget(); tf=QFormLayout(traffic_page)
        self.traffic_enabled=QCheckBox("启用时空业务模型"); self.traffic_enabled.setChecked(True); tf.addRow(self.traffic_enabled)
        traffic_path_row=QHBoxLayout(); self.traffic_grid_path=QLineEdit(str(DEFAULT_TRAFFIC_GRID)); self.traffic_grid_path.setPlaceholderText("留空使用中国区域规则演示网格")
        self.browse_traffic_btn=QPushButton("选择CSV"); self.browse_traffic_btn.clicked.connect(self._browse_traffic_grid); traffic_path_row.addWidget(self.traffic_grid_path); traffic_path_row.addWidget(self.browse_traffic_btn); tf.addRow("业务网格",traffic_path_row)
        self.traffic_dense_percentile=self._spin(0,100,95,1," %"); tf.addRow("高密业务百分位",self.traffic_dense_percentile)
        self.traffic_dense_max_satellites=self._spin(4,10,4);self.traffic_dense_max_satellites.setToolTip("热点区域至少使用4颗可见接入卫星分担聚合业务，避免单颗业务卫星在最多2个馈电出口约束下形成结构性拥塞");tf.addRow("热点簇协同卫星上限",self.traffic_dense_max_satellites)
        self.traffic_cluster_radius=self._spin(1,2000,180,1," km"); tf.addRow("热点相邻判定距离",self.traffic_cluster_radius)
        self.traffic_total_gbps=self._spin(0.001,1000000,500,1," Gb/s")
        self.traffic_total_gbps.setToolTip("业务规模基准Dref。各网格人口占比乘以Dref和当地时段系数，因此某一时刻的全网总业务量通常不等于Dref。")
        tf.addRow("业务规模基准 Dref",self.traffic_total_gbps)
        self.traffic_landing_percent=self._spin(0,100,30,1," %")
        self.traffic_landing_percent.setToolTip("接入业务中需要经卫星汇聚至信关站并进入地面网络的比例。默认30%，用于使默认业务规模与默认信关站出口能力处于同一量级。")
        tf.addRow("业务落地比例",self.traffic_landing_percent)
        self.traffic_random_sigma=self._spin(0,2,0,2); tf.addRow("随机业务波动σ",self.traffic_random_sigma)
        self.traffic_outside_china_percent=self._spin(0,100,0,1," %"); tf.addRow("中国以外业务量",self.traffic_outside_china_percent)
        self.traffic_outside_china_percent.setToolTip("相对于各境外网格原始人口业务权重的比例；0%表示境外业务不参与计算和显示")
        self.priority_region_table=QTableWidget(0,4);self.priority_region_table.setHorizontalHeaderLabels(["任务区域","纬度","经度","范围 km"]);self.priority_region_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch);self.priority_region_table.setMaximumHeight(150)
        self.priority_region_table.setToolTip("重点区域按中心坐标和半径构成圆形范围；新疆默认值为近似覆盖，可按业务范围调整。")
        self._append_priority_region("新疆重点区域",42.0,85.0,900)
        priority_buttons=QHBoxLayout();self.add_priority_region_btn=QPushButton("增加重点区域");self.remove_priority_region_btn=QPushButton("删除重点区域");self.add_priority_region_btn.clicked.connect(lambda:self._append_priority_region("重点区域",39.9,116.4,300));self.remove_priority_region_btn.clicked.connect(self._remove_priority_region);priority_buttons.addWidget(self.add_priority_region_btn);priority_buttons.addWidget(self.remove_priority_region_btn)
        tf.addRow("重点任务区域",self.priority_region_table);tf.addRow(priority_buttons)
        traffic_note=QLabel("业务规模基准不是每帧固定总速率；时段系数会改变实际业务。落地需求 = 当前接入业务 × 落地比例。修改此页后需重新生成基础数据并重新规划。")
        traffic_note.setWordWrap(True); tf.addRow(traffic_note); self.property_stack.addWidget(traffic_page)
        planning_page=QWidget();planning_page_layout=QVBoxLayout(planning_page);planning_page_layout.setContentsMargins(8,8,8,8);planning_page_layout.setSpacing(10)
        planning_parameters=QGroupBox("规划参数");pf=QFormLayout(planning_parameters)
        self.planning_enabled=QCheckBox("启用离线规划与实时调度");self.planning_enabled.setChecked(True);pf.addRow(self.planning_enabled)
        self.planning_period=self._spin(300,300,300,0," s");self.planning_period.setToolTip("离线方案固定每5分钟生效一次，30分钟场景共6个时段")
        self.planning_horizon=self._spin(1800,1800,1800,0," s");self.planning_horizon.setToolTip("每次离线规划覆盖完整30分钟场景")
        self.planning_slot=self._spin(300,300,300,0," s");self.planning_slot.hide()
        self.planning_capacity_reserve=self._spin(0,100,20,1," %");self.planning_capacity_reserve.setToolTip("规划容量目标 = 落地业务 × (1 + 容量余量)")
        pf.addRow("方案生效间隔",self.planning_period);pf.addRow("方案覆盖时长",self.planning_horizon);pf.addRow("规划容量余量",self.planning_capacity_reserve)
        self.planning_strategy=QComboBox();self.planning_strategy.addItem("容量可行优先时延", "legacy");self.planning_strategy.addItem("容量时延加权", "window_margin")
        self.planning_strategy.setToolTip("容量可行优先时延：先筛容量可行候选，再选低时延；没有可行候选时优先补容量。容量时延加权：对五分钟窗口容量与路径时延加权评分。")
        self.planning_capacity_weight=self._spin(0,100,50,0," %")
        self.planning_capacity_weight.setToolTip("容量权重；时延权重自动取剩余百分比。容量使用五分钟采样平均速率，链路可用性仍按最低速率检查。")
        self.planning_capacity_weight.setEnabled(False)
        self.planning_strategy.currentIndexChanged.connect(lambda: self.planning_capacity_weight.setEnabled(self.planning_strategy.currentData()=="window_margin"))
        pf.addRow("选星策略",self.planning_strategy);pf.addRow("容量权重",self.planning_capacity_weight)
        planning_page_layout.addWidget(planning_parameters);planning_page_layout.addStretch();self.property_stack.addWidget(planning_page)
        scheduling_page=QWidget();scheduling_page_layout=QVBoxLayout(scheduling_page);scheduling_page_layout.setContentsMargins(8,8,8,8);scheduling_page_layout.setSpacing(10)
        scheduling_parameters=QGroupBox("调度参数");sf=QFormLayout(scheduling_parameters)
        self.fast_evaluation_period=self._spin(.1,60,1,1," s");self.fast_adjust_period=self._spin(1,1,1,1," s")
        self.route_update_period=self._spin(1,3600,15,1," s")
        self.switching_loss=self._spin(0,100,0,1," %");self.switching_loss.hide()
        self.offline_rain_grade=QComboBox();self.offline_rain_grade.addItem("请先生成离线规划方案",None);self.offline_rain_grade.setEnabled(False)
        self.offline_rain_grade.setToolTip("从本次生成的规划方案中选择执行工况；每种工况包含6个五分钟方案，仿真按时间自动切换")
        self.offline_rain_grade.currentIndexChanged.connect(self._select_offline_rain_grade)
        self.fast_evaluation_period.editingFinished.connect(self._apply_scheduling_parameters)
        self.fast_adjust_period.editingFinished.connect(self._apply_scheduling_parameters)
        sf.addRow("选择规划方案",self.offline_rain_grade)
        sf.addRow("链路状态检测周期",self.fast_evaluation_period);sf.addRow("不可达业务寻址周期",self.fast_adjust_period)
        scheduling_page_layout.addWidget(scheduling_parameters);scheduling_page_layout.addStretch();self.property_stack.addWidget(scheduling_page)
        link_parameters=QGroupBox("馈电链路能力");lf=QFormLayout(link_parameters)
        self.feeder_bandwidth=self._spin(1,100000,500,1," MHz");self.snr_margin=self._spin(0,20,1,1," dB")
        self.feeder_frequency=self._spin(1,100,39,1," GHz")
        self.feeder_rolloff=self._spin(0,.50,.20,2)
        lf.addRow("馈电下行频率（参考值）",self.feeder_frequency)
        lf.addRow("成形滚降系数",self.feeder_rolloff)
        lf.addRow("单信道占用带宽",self.feeder_bandwidth);lf.addRow("Es/N0保护余量",self.snr_margin)
        modcod_path_row=QHBoxLayout();self.modcod_table_path=QLineEdit();self.modcod_table_path.setPlaceholderText("留空使用默认表")
        self.browse_modcod_btn=QPushButton("选择CSV");self.browse_modcod_btn.clicked.connect(self._browse_modcod_table);modcod_path_row.addWidget(self.modcod_table_path);modcod_path_row.addWidget(self.browse_modcod_btn);lf.addRow("Es/N0—速率表",modcod_path_row)
        modcod_text="\n".join(f"{row.name}：Es/N0≥{row.minimum_snr_db:g} dB，η={row.spectral_efficiency:g}" for row in DEFAULT_MODCOD_TABLE)
        modcod_label=QLabel("当前默认调制编码表\n"+modcod_text);modcod_label.setWordWrap(True);modcod_label.setStyleSheet("background:#0a2131;border:1px solid #1a617c;padding:8px;color:#bfe8f5")
        lf.addRow(modcod_label)
        budget_note=QLabel("参考下行预算：Es/N0 = 卫星EIRP + 信关站G/T − 自由空间损耗 − 接收系统损失 − 雨衰 − 干扰 + 228.6 − 10log10(符号率Hz)。假设理想跟踪，指向损失为0；卫星EIRP需用设备实测值校准。")
        budget_note.setWordWrap(True);lf.addRow(budget_note);gl.insertWidget(3,link_parameters)
        self.lat_limit=self._spin(0,90,90,1,"°"); self.max_isl=self._spin(500,10000,2000,0," km")
        self.max_isl.setKeyboardTracking(False)
        self.max_isl.setToolTip("星间链路的距离上限。相邻卫星间距离超过此值时，该ISL不参与规划与实时拓扑；默认2000 km。")
        self.lat_limit.setToolTip("异轨ISL两端卫星的纬度上限；90°表示不因纬度额外截断。实际链路仍受距离与地球遮挡约束。")
        self._lat_limit_auto=True;self._setting_lat_limit=False
        self.lat_limit.valueChanged.connect(self._latitude_limit_changed)
        self.inclination.valueChanged.connect(self._sync_default_latitude_limit)
        self._sync_default_latitude_limit()
        isl_page=QWidget();isl_form=QFormLayout(isl_page)
        isl_form.addRow("链路构型",self.isl_pattern);isl_form.addRow("异轨ISL纬度上限",self.lat_limit);isl_form.addRow("星间链路距离上限",self.max_isl)
        isl_note=QLabel("同轨面连接前后邻星；异轨面连接相邻轨道面。每帧按纬度、星间距离和地球遮挡重新判断链路。距离上限是有效链路约束，不是画线长度设置。")
        isl_note.setWordWrap(True);isl_form.addRow(isl_note);self.property_stack.insertWidget(1,isl_page)
        rain_page=QWidget();rain_form=QFormLayout(rain_page)
        self.rain_enabled=QCheckBox("启用雨衰模型");self.rain_enabled.setChecked(True)
        self.rain_source=QComboBox();self.rain_source.addItems(["按地理位置自动生成","手动雨衰序列","外部气象文件"])
        self.rain_sense=QComboBox();self.rain_sense.addItems(["无感知 σ=0.50","站级感知 σ=0.35","天线级感知 σ=0.20","天线级 + CQI σ=0.12"]);self.rain_sense.setCurrentIndex(2)
        self.rain_forecast=self._spin(0,48,2,1," h");self.rain_p90=QCheckBox("规划采用P90预测上界");self.rain_p90.setChecked(True)
        rain_form.addRow(self.rain_enabled);rain_form.addRow("雨衰数据来源",self.rain_source);rain_form.addRow("感知精度",self.rain_sense);rain_form.addRow("预报提前期",self.rain_forecast);rain_form.addRow(self.rain_p90)
        rain_note=QLabel("离线规划与实时调度使用所选雨衰等级对应的同一空间雨场；链路预算仍按实时仰角和斜距更新。")
        rain_note.setWordWrap(True);rain_form.addRow(rain_note);self.property_stack.insertWidget(4,rain_page)
        # 地图显示已迁入 Cesium；该隐藏页仍保存 Python 侧默认值，必须有持久父对象。
        display=QWidget(self); display.hide(); self._display_defaults_page=display; df=QFormLayout(display)
        self.center_plane=self._spin(1,500,1); self.center_plane.setToolTip("轨道面从1开始编号")
        self.center_plane.setMaximum(self.planes.value())
        self.planes.valueChanged.connect(self.center_plane.setMaximum)
        self.display_planes=QComboBox(); self.display_planes.addItems(["邻近3个轨道面", "邻近5个轨道面", "全部轨道面"])
        self.center_plane.valueChanged.connect(self._plane_filter_changed)
        self.display_planes.currentIndexChanged.connect(self._plane_filter_changed)
        df.addRow("中心轨道面",self.center_plane); df.addRow("显示范围",self.display_planes)
        self.show_sat=QCheckBox("卫星"); self.show_same=QCheckBox("同轨ISL"); self.show_cross=QCheckBox("异轨ISL"); self.show_gsl=QCheckBox("地球站—可见卫星 GSL连线"); self.show_motion=QCheckBox("卫星轨迹方向箭头"); self.show_labels=QCheckBox("轨道/卫星编号（放大后显示）")
        for w in (self.show_sat,self.show_same,self.show_cross,self.show_gsl,self.show_motion,self.show_labels):
            w.setChecked(True); w.toggled.connect(self._layer_changed); df.addRow(w)
        # 显示选项改由 Cesium 地图右上角直接控制；控件保留为后台默认值。
        self.property_buttons=[]
        for index,title in enumerate(("星座建模","星间链路建模","信关站建模","业务建模","雨衰建模","离线规划","实时调度")):
            button=QToolButton(); button.setObjectName("verticalPropertyTab"); button.setText("\n".join(title)); button.setCheckable(True); button.setFixedWidth(42); button.setMinimumHeight(92); button.clicked.connect(lambda checked,i=index:self._select_property_drawer(i)); self.property_rail_layout.addWidget(button); self.property_buttons.append(button)
        self.property_rail_layout.addStretch(); self._active_property_index=0
        self.display_planes.setCurrentIndex(2)
        self.scenario_edit_widgets=[self.source,self.tle_path,self.browse_tle_btn,self.walker_mode,self.shell_count,self.shell_table,self.sat_count,self.planes,self.walker_layout,self.altitude,self.inclination,self.orbit_shape,self.eccentricity,self.arg_perigee,self.satellite_antennas,self.satellite_channels,self.satellite_channel_capacity,self.geo_interference,self.geo_satellite_count,self.geo_center_longitude,self.geo_longitude_spacing,self.geo_altitude,self.gso_i_over_n,self.gso_protection_angle,self.duration,self.step,self.isl_pattern,self.station_preset,self.add_station_btn,self.station_table,self.remove_station_btn,self.gateway_planning_enabled,self.gateway_planning_mode,self.gateway_max_sites,self.gateway_redundancy,self.gateway_max_antennas,self.elevation,self.gateway_eirp,self.gateway_gt,self.gateway_array_loss,self.satellite_eirp,self.feeder_frequency,self.feeder_rolloff,self.lat_limit,self.max_isl,self.traffic_enabled,self.traffic_grid_path,self.browse_traffic_btn,self.traffic_dense_percentile,self.traffic_dense_max_satellites,self.traffic_cluster_radius,self.traffic_total_gbps,self.traffic_landing_percent,self.traffic_random_sigma,self.traffic_outside_china_percent,self.priority_region_table,self.add_priority_region_btn,self.remove_priority_region_btn,self.rain_enabled,self.rain_source,self.rain_sense,self.rain_forecast,self.rain_p90,self.feeder_bandwidth,self.snr_margin,self.modcod_table_path,self.browse_modcod_btn]
        compute_label=QLabel("数据准备进度 / 当前仿真时间位置")
        compute_label.setToolTip("生成阶段显示TLE、轨道对象和ISL候选的准备进度；播放后显示当前时间帧位置")
        self.compute_progress=QProgressBar(); self.compute_progress.setMaximumWidth(180)
        transfer_label=QLabel("当前帧送显进度（Python → Cesium）")
        transfer_label.setToolTip("当前时间帧的卫星、ISL和GSL数据分块传入地图的进度，不是网络下载进度")
        self.transfer_progress=QProgressBar(); self.transfer_progress.setMaximumWidth(150); self.status=QLabel("等待开始");self.status.setWordWrap(False);self.status.setSizePolicy(QSizePolicy.Policy.Ignored,QSizePolicy.Policy.Fixed)
        right=QWidget(); right_layout=QVBoxLayout(right); right_layout.setContentsMargins(0,0,0,0); right_layout.setSpacing(4)
        self.web=QWebEngineView(); self.web.page().setBackgroundColor(QColor("#07131f")); right_layout.addWidget(self.web,1)
        playback=QHBoxLayout()
        self.timeline=SeekSlider(Qt.Orientation.Horizontal); self.timeline.setEnabled(False); self.timeline.setRange(0,0);self.timeline.setStyleSheet("QSlider::groove:horizontal{height:5px;background:#17384a;border-radius:2px}QSlider::sub-page:horizontal{background:#00cfe8;border-radius:2px}QSlider::handle:horizontal{width:16px;margin:-6px 0;background:#ff3b4f;border:2px solid #ffd5da;border-radius:8px}")
        self.timeline.sliderPressed.connect(self._timeline_pressed);self.timeline.sliderMoved.connect(self._timeline_moved); self.timeline.sliderReleased.connect(self._timeline_released)
        self.timeline_label=QLabel("t=0 s"); self.timeline_label.setMinimumWidth(82)
        self.elapsed_label=QLabel("准备耗时 00:00:00");self.elapsed_label.setMinimumWidth(145)
        self.absolute_time_label=QLabel("UTC：--"); self.absolute_time_label.setMinimumWidth(225)
        self.speed.currentTextChanged.connect(self._speed_changed)
        playback.addWidget(QLabel("时间")); playback.addWidget(self.timeline,1)
        playback.addWidget(self.elapsed_label);playback.addWidget(self.timeline_label); playback.addWidget(self.absolute_time_label)
        right_layout.addLayout(playback)
        view_tabs=CenteredTabWidget(); self.view_tabs=view_tabs; view_tabs.setObjectName("mainWorkspaceTabs")
        view_tabs.segmentWidget().setObjectName("mainSegmentGroup")
        planning_page=QWidget();planning_layout=QVBoxLayout(planning_page);planning_layout.setContentsMargins(0,0,0,0)
        self.planning_tabs=CenteredTabWidget();self.planning_tabs.setObjectName("planningWorkspaceTabs")
        self.planning_tabs.segmentWidget().setObjectName("planningSegmentGroup")
        planning_layout.addWidget(self.planning_tabs)
        planning_map=QWidget();self.planning_map_layout=QVBoxLayout(planning_map);self.planning_map_layout.setContentsMargins(0,0,0,0)
        input_map=QWidget();self.input_map_layout=QVBoxLayout(input_map);self.input_map_layout.setContentsMargins(0,0,0,0)
        self.planning_results=PlanningResultsWidget()
        self.planning_resources=PlanningResourceWidget()
        self.planning_evaluation=PlanningEvaluationWidget();self.result_chart=self.planning_evaluation.chart
        self._syncing_gateway_scope=False;self.selected_result_gateway=None
        for selector in (self.planning_results.gateway_scope,self.planning_resources.gateway_scope,self.planning_evaluation.gateway_scope):
            selector.currentChanged.connect(lambda _index,source=selector:self._sync_result_gateway_scope(source))
        self.planning_tabs.addTab(planning_map,"调度态势")
        self.planning_tabs.addTab(self.planning_results,"规划方案")
        self.planning_tabs.addTab(self.planning_resources,"调度执行")
        self.planning_tabs.addTab(self.planning_evaluation,"效能评估")
        self.planning_tabs.addTab(input_map,"拓扑与业务")
        self._workspace_groups={"constellation":[],"calculation":[],"simulation":[]}
        self.base_parameter_status=[]
        base_titles=("星座模型","星间链路","信关站","业务模型","雨衰模型")
        base_page=QWidget();base_column=QVBoxLayout(base_page);base_column.setContentsMargins(8,8,8,8);base_column.setSpacing(8)
        self.base_tabs=GatewayScopeTabs();self.base_tabs.setExpanding(True);base_column.addWidget(self.base_tabs)
        self.base_stack=QStackedWidget();base_column.addWidget(self.base_stack,1)
        for title in base_titles:
            page=self.property_stack.widget(0);self.property_stack.removeWidget(page)
            scroll=QScrollArea();scroll.setWidgetResizable(True);scroll.setFrameShape(QScrollArea.Shape.NoFrame)
            scroll.setWidget(page);self.base_stack.addWidget(scroll);self.base_tabs.addItem(title)
        self.base_tabs.currentChanged.connect(self.base_stack.setCurrentIndex)
        self.base_tabs.currentChanged.connect(lambda index:setattr(self,"_active_property_index",index))
        status=QLabel();status.setWordWrap(True);status.setMinimumHeight(48)
        base_column.addWidget(status);self.base_parameter_status.append(status)
        base_index=view_tabs.addTab(base_page,"基础参数");self._workspace_groups["constellation"].append(base_index)
        self._property_workspace_indexes=[base_index]*5
        for title in ("规划参数","调度参数"):
            page=self.property_stack.widget(0);self.property_stack.removeWidget(page)
            scroll=QScrollArea();scroll.setWidgetResizable(True);scroll.setFrameShape(QScrollArea.Shape.NoFrame)
            scroll.setWidget(page)
            index=view_tabs.addTab(scroll,title);self._workspace_groups["constellation"].append(index);self._property_workspace_indexes.append(index)
        self._result_workspace_index=view_tabs.addTab(planning_page,"仿真结果")
        # 组内索引仍以结果页为默认入口；屏幕上的导航顺序则先模型与参数，
        # 最后显示仿真结果。
        self._workspace_groups["constellation"]=[self._result_workspace_index]+self._workspace_groups["constellation"]
        try:
            for title,page in create_calculation_pages(view_tabs):
                self._workspace_groups["calculation"].append(view_tabs.addTab(page,title))
            for title,page in create_simulation_pages(view_tabs):
                self._workspace_groups["simulation"].append(view_tabs.addTab(page,title))
        except Exception as exc:
            for group in ("calculation","simulation"):
                error_page=QLabel(f"链路工程加载失败\n{type(exc).__name__}: {exc}");error_page.setAlignment(Qt.AlignmentFlag.AlignCenter);error_page.setStyleSheet("color:#ff7b89;font-size:14px")
                self._workspace_groups[group].append(view_tabs.addTab(error_page,"加载失败"))
        view_tabs.setNavigationVisible(False)
        self.planning_tabs.setNavigationVisible(False)
        self.workspace_context_nav=QStackedWidget();self.workspace_context_nav.setObjectName("workspaceContextNav");self.workspace_context_nav.setMaximumHeight(34)
        planning_context=QWidget();planning_context.setObjectName("contextNavPage");planning_context_layout=QHBoxLayout(planning_context);planning_context_layout.setContentsMargins(0,0,0,0);planning_context_layout.addWidget(self.planning_tabs.segmentWidget())
        self.workspace_context_nav.addWidget(planning_context)
        view_tabs.addNavigationWidget(self.workspace_context_nav)
        self._secondary_workspace_buttons={}
        self._secondary_button_group=QButtonGroup(self);self._secondary_button_group.setExclusive(True)
        for group,indexes in self._workspace_groups.items():
            group_buttons=[]
            for local_index,page_index in enumerate(indexes):
                title=view_tabs._buttons[page_index].text()
                button=QToolButton();button.setObjectName("secondaryProjectTab");button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly);button.setText(title);button.setCheckable(True);button.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Fixed);button.setMinimumHeight(34)
                button.clicked.connect(lambda checked,k=group,i=local_index:self._switch_primary_workspace(k,i,True))
                self._secondary_button_group.addButton(button);self._secondary_nav_layouts[group].addWidget(button);group_buttons.append(button)
            self._secondary_workspace_buttons[group]=group_buttons
        self.map_workspace=right;self.planning_map_layout.addWidget(self.map_workspace)
        view_tabs.currentChanged.connect(self._central_workspace_changed)
        self.planning_tabs.currentChanged.connect(self._planning_subtab_changed)
        panel.hide();panel.setFixedWidth(0)
        splitter.addWidget(project_panel); splitter.addWidget(view_tabs); splitter.addWidget(panel)
        splitter.setStretchFactor(0,0); splitter.setStretchFactor(1,1); splitter.setStretchFactor(2,0)
        self.busy_overlay=BusyOverlay(view_tabs);view_tabs.busy_overlay=self.busy_overlay
        self.project_tree.itemClicked.connect(self._project_item_selected)

        import_scene=self._action("导入场景",QStyle.StandardPixmap.SP_ArrowDown,self._import_scene_config); export_scene=self._action("导出场景",QStyle.StandardPixmap.SP_ArrowUp,self._export_scene_config)
        prepare_action=self._action("生成基础数据",QStyle.StandardPixmap.SP_DriveHDIcon,self.start_simulation);self.prepare_btn=prepare_action
        prepare_action.setToolTip("按当前场景参数生成或复用轨道、业务基础数据。修改星座、信关站、业务或链路参数后，需要重新生成。")
        offline_plan_action=self._action("离线规划",QStyle.StandardPixmap.SP_FileDialogDetailedView,self._run_offline_planning);self.offline_plan_btn=offline_plan_action;self.offline_plan_btn.setEnabled(False)
        offline_plan_action.setToolTip("使用当前已准备的基础数据生成或复用18个离线方案（6个五分钟周期 × 3种雨况）；基础数据准备完成后可点击。")
        run_action=self._action("开始仿真",QStyle.StandardPixmap.SP_MediaPlay,self._primary_action); self.start_btn=run_action;self.start_btn.setEnabled(False)
        run_action.setToolTip("按「调度参数」中选择的雨况方案运行仿真；需先完成基础数据和离线规划。运行时每帧更新链路状态。")
        run_all_action=self._action("一键运行",QStyle.StandardPixmap.SP_MediaPlay,self._run_all_stages);self.run_all_btn=run_all_action;self.run_all_btn.setToolTip("按当前场景参数依次生成基础数据、离线规划方案并启动仿真；仿真使用「调度参数」里选择的雨况方案。")
        pause_action=self._action("暂停",QStyle.StandardPixmap.SP_MediaPause,self._pause_simulation);self.pause_btn=pause_action;self.pause_btn.setEnabled(False)
        stop_action=self._action("结束",QStyle.StandardPixmap.SP_MediaStop,self.stop_simulation); self.stop_btn=stop_action; self.stop_btn.setEnabled(False)
        save_layout=self._action("保存当前布局",QStyle.StandardPixmap.SP_DialogSaveButton,self._save_layout); restore_layout=self._action("加载保存布局",QStyle.StandardPixmap.SP_DialogOpenButton,self._restore_layout); reset_layout=self._action("恢复默认布局",QStyle.StandardPixmap.SP_BrowserReload,self._reset_layout)
        save_simulation_data_action=self._action("保存仿真数据",QStyle.StandardPixmap.SP_DialogSaveButton,self._save_simulation_data);save_simulation_data_action.setToolTip("导出当前调度执行和效能评估页面的摘要、表格与趋势图")
        export_action=self._action("导出基础数据",QStyle.StandardPixmap.SP_DialogSaveButton,self.export_basic_data); export_tables_action=self._action("导出全部表格",QStyle.StandardPixmap.SP_DialogSaveButton,self._export_all_result_tables); about_action=self._action("关于",QStyle.StandardPixmap.SP_MessageBoxInformation,lambda:QMessageBox.information(self,"关于","星地馈电链路与星座仿真验证系统\n链路计算、链路仿真与星座仿真一体化工作台"))
        scene_menu=self.menuBar().addMenu("场景"); scene_menu.addActions([import_scene,export_scene])
        simulation_menu=self.menuBar().addMenu("仿真"); simulation_menu.addActions([prepare_action,offline_plan_action,run_action,run_all_action,pause_action,stop_action])
        data_menu=self.menuBar().addMenu("数据"); data_menu.addAction(save_simulation_data_action); data_menu.addSeparator(); data_menu.addActions([export_action,export_tables_action])
        view_menu=self.menuBar().addMenu("视图"); view_menu.addActions([save_layout,restore_layout,reset_layout])
        self.fullscreen_action=QAction("全屏显示",self);self.fullscreen_action.setCheckable(True);self.fullscreen_action.setShortcut("F11");self.fullscreen_action.triggered.connect(self.set_fullscreen);view_menu.addAction(self.fullscreen_action)
        self.exit_fullscreen_action=QAction("退出全屏",self);self.exit_fullscreen_action.setShortcut("Esc");self.exit_fullscreen_action.setEnabled(False);self.exit_fullscreen_action.triggered.connect(self.exit_fullscreen);self.addAction(self.exit_fullscreen_action)
        self.left_panel_action=self._action("工程抽屉",QStyle.StandardPixmap.SP_DirIcon,lambda:self._toggle_panel(self.project_panel,300)); self.left_panel_action.setCheckable(True); self.left_panel_action.setChecked(False)
        self.right_panel_action=self._action("参数页",QStyle.StandardPixmap.SP_FileDialogContentsView,lambda:self._switch_primary_workspace("constellation",1)); self.right_panel_action.setVisible(False)
        view_menu.addSeparator(); view_menu.addAction(self.left_panel_action)
        help_menu=self.menuBar().addMenu("帮助"); help_menu.addAction(about_action)
        toolbar=self.addToolBar("常用命令"); toolbar.setObjectName("mainToolbar"); toolbar.setMovable(True); toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        icon_dir=Path(__file__).with_name("assets")
        for action,name in ((import_scene,"tle-import.svg"),(export_scene,"export.svg"),(run_all_action,"run.svg"),(prepare_action,"prepare-data.svg"),(offline_plan_action,"offline-plan.svg"),(run_action,"run.svg"),(pause_action,"pause.svg"),(stop_action,"stop.svg"),(save_simulation_data_action,"export.svg"),(export_action,"export.svg"),(export_tables_action,"export.svg")):
            icon_file=icon_dir/name
            if icon_file.exists():action.setIcon(QIcon(str(icon_file)))
        toolbar.addActions([prepare_action,offline_plan_action,run_action,run_all_action]);toolbar.addSeparator()
        toolbar.addActions([pause_action,stop_action]);toolbar.addSeparator();toolbar.addAction(save_simulation_data_action);toolbar.addSeparator()
        self.duration.setMaximumWidth(92);self.step.setMaximumWidth(82)
        toolbar.addWidget(QLabel("时长"));toolbar.addWidget(self.duration);toolbar.addWidget(QLabel("步进"));toolbar.addWidget(self.step);toolbar.addWidget(QLabel("倍速"));toolbar.addWidget(self.speed);toolbar.addSeparator()
        status_bar=self.statusBar();status_bar.setSizeGripEnabled(False);status_bar.setFixedHeight(36)
        self.state_badge=QLabel("● 就绪");self.cache_status_label=QLabel("缓存 --");self.cache_status_label.setMaximumWidth(145);self.cache_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter);self.cache_status_label.setStyleSheet("QLabel{color:#ff6675;background:#35131b;border:1px solid #ff3b4f;border-radius:3px;padding:3px 8px;font-weight:700}");self.cache_status_label.setToolTip("悬停查看各模型缓存状态"); self.resource_label=QLabel("资源 CPU -- / 内存 --"); self.alert_label=QLabel("告警 0")
        for widget in (self.state_badge,self.cache_status_label,self.resource_label,self.alert_label): widget.setObjectName("statusSegment")
        status_bar.addWidget(self.state_badge); status_bar.addWidget(self.status,1);status_bar.addPermanentWidget(self.cache_status_label); status_bar.addPermanentWidget(QLabel("计算")); status_bar.addPermanentWidget(self.compute_progress); status_bar.addPermanentWidget(QLabel("传输")); status_bar.addPermanentWidget(self.transfer_progress); status_bar.addPermanentWidget(self.resource_label); status_bar.addPermanentWidget(self.alert_label)
        for name in ("佳木斯","北京","喀什","拉萨","三亚"):self._append_station(name,*PRESET_STATIONS[name])
        for widget in (self.source,self.sat_count,self.planes,self.duration,self.step):
            signal=widget.currentIndexChanged if isinstance(widget,QComboBox) else widget.valueChanged
            signal.connect(self._refresh_scene_preview)
        for widget in self.scenario_edit_widgets:
            for signal_name in ("valueChanged","currentIndexChanged","toggled","textChanged","itemChanged"):
                signal=getattr(widget,signal_name,None)
                if signal is not None:signal.connect(self._refresh_parameter_actions)
        self.station_table.itemChanged.connect(self._station_item_changed)
        self._gateway_mode_changed(self.gateway_planning_mode.currentIndex())
        self._refresh_scene_preview()
        self._switch_primary_workspace("constellation")

    def _primary_workspace_clicked(self,key):
        if getattr(self,"_active_primary_workspace",None)==key and self._secondary_nav_containers[key].isVisible():
            self._set_secondary_navigation(False);return
        self._switch_primary_workspace(key,0,True)

    def _set_secondary_navigation(self,expanded):
        active=getattr(self,"_active_primary_workspace",None)
        for key,container in self._secondary_nav_containers.items():container.setVisible(expanded and key==active)
        width=168 if expanded else 48
        self.project_rail.setFixedWidth(width)
        if not self._drawer_expanded(self.project_panel):self.project_panel.setFixedWidth(width)
        for key,button in zip(("calculation","simulation","constellation"),self.project_buttons):
            button.setChecked(key==getattr(self,"_active_primary_workspace",None))
            if expanded:button.setText(("▼  " if key==active else "▶  ")+self._project_titles[key])
            else:button.setText("\n".join(self._project_titles[key]))
            button.setFixedWidth(width-6 if expanded else 42);button.setMinimumHeight(44 if expanded else 96)
        self._enforce_splitter_layout()
    def _switch_primary_workspace(self,key,index_in_group=0,expand_navigation=False):
        groups=getattr(self,"_workspace_groups",{})
        indexes=groups.get(key,[])
        if not indexes:return
        self._active_primary_workspace=key
        for group_indexes in groups.values():
            for index in group_indexes:
                if index < len(self.view_tabs._buttons):self.view_tabs._buttons[index].setVisible(index in indexes)
        target=indexes[min(max(0,index_in_group),len(indexes)-1)]
        self.view_tabs.setCurrentIndex(target)
        if key in self._secondary_workspace_buttons and self._secondary_workspace_buttons[key]:
            self._secondary_workspace_buttons[key][min(max(0,index_in_group),len(self._secondary_workspace_buttons[key])-1)].setChecked(True)
        self.project_drawer.hide();self.project_panel.setProperty("drawerExpanded",False)
        self._set_secondary_navigation(expand_navigation)
        self.workspace_context_nav.setVisible(key=="constellation" and target==getattr(self,"_result_workspace_index",-1))
        for pos,button in enumerate(self.project_buttons):button.setChecked((key==("calculation","simulation","constellation")[pos]))
        self.project_drawer.hide();self.project_panel.setProperty("drawerExpanded",False);self._enforce_splitter_layout()
        self.view_tabs._position_overlay_navigation()

    def _central_workspace_changed(self,index):
        if index in getattr(self,"_property_workspace_indexes",[])[:5]:self._refresh_parameter_actions()
        is_planning=index==getattr(self,"_result_workspace_index",-1)
        self.workspace_context_nav.setCurrentIndex(0);self.workspace_context_nav.setVisible(is_planning);self.view_tabs._position_overlay_navigation()
        if is_planning:
            self._planning_subtab_changed(self.planning_tabs.currentIndex())
            self._sync_worker_display_options()
            QTimer.singleShot(0,self._refresh_cesium_view)

    def _planning_subtab_changed(self,index):
        if index==0:
            self.planning_map_layout.addWidget(self.map_workspace)
            if self.web_ready:self.web.page().runJavaScript("setViewPage('planning')")
            for delay in (0,100,300):QTimer.singleShot(delay,self._refresh_cesium_view)
        elif index==2 and self.current_full_frame and self.active_config:
            self.planning_resources.update_frame(self.current_full_frame,[x["name"] for x in self.active_config.stations])
        elif index==4:
            self.input_map_layout.addWidget(self.map_workspace)
            if self.web_ready:self.web.page().runJavaScript(f"setViewPage('{self.input_view_mode}')")
            for delay in (0,100,300):QTimer.singleShot(delay,self._refresh_cesium_view)
        self._sync_worker_display_options()

    def _sync_result_gateway_scope(self,source):
        if self._syncing_gateway_scope:return
        self._syncing_gateway_scope=True
        try:
            gateway=source.currentData();self.selected_result_gateway=gateway
            for selector in (self.planning_results.gateway_scope,self.planning_resources.gateway_scope,self.planning_evaluation.gateway_scope):
                if selector is source:continue
                index=selector.findData(gateway)
                if index>=0 and index!=selector.currentIndex():selector.setCurrentIndex(index)
        finally:self._syncing_gateway_scope=False

    @Slot(str)
    def _input_view_changed(self,mode):
        if mode not in ("topology","business") or mode==self.input_view_mode:return
        self.input_view_mode=mode;self._sync_worker_display_options()

    def _project_item_selected(self,item,_column):
        tab_index=item.data(0,Qt.ItemDataRole.UserRole)
        if isinstance(tab_index,int):
            self._show_property_section(tab_index); self.property_title.setText(f"属性面板 · {item.text(0)}")

    def _show_property_section(self,index):
        self._active_property_index=index
        if hasattr(self,"_property_workspace_indexes") and 0<=index<len(self._property_workspace_indexes):
            target=self._property_workspace_indexes[index]
            local_index=self._workspace_groups["constellation"].index(target)
            self._switch_primary_workspace("constellation",local_index)
            if index<5:
                self.base_tabs.setCurrentIndex(index)
                self.base_stack.setCurrentIndex(index)

    def _select_project_drawer(self,index):
        if not self._drawer_expanded(self.project_panel):
            self._active_project_index=index; self._focus_project_target(); self._set_project_drawer(True)
        elif self._active_project_index == index:
            self._set_project_drawer(False)
        else:
            self._active_project_index=index; self._focus_project_target(); self._update_project_buttons(True)

    def _focus_project_target(self):
        target=self._project_targets[self._active_project_index]; self.project_tree.setCurrentItem(target); target.setExpanded(True)
        parent=target.parent()
        while parent is not None:parent.setExpanded(True); parent=parent.parent()

    def _update_project_buttons(self,expanded):
        for key,button in zip(("calculation","simulation","constellation"),self.project_buttons):
            button.setChecked(key==getattr(self,"_active_primary_workspace",None))

    def _set_project_drawer(self,expanded):
        if expanded:self._focus_project_target()
        self._animate_side_drawer(self.project_panel,expanded,300,self.left_panel_action); self._update_project_buttons(expanded)

    def _select_property_drawer(self,index):
        self._show_property_section(index)

    def _update_property_buttons(self,expanded):
        for i,button in enumerate(self.property_buttons):button.setChecked(expanded and i==self._active_property_index)

    def _set_property_drawer(self,expanded):
        self.property_panel.setProperty("drawerExpanded",False);self.property_drawer.hide();self.property_panel.hide();self.property_panel.setFixedWidth(0)
        self._enforce_splitter_layout()

    def _drawer_expanded(self,panel):
        return bool(panel.property("drawerExpanded"))

    def _animate_side_drawer(self,panel,expanded,full_width,action):
        drawer=self.project_drawer if panel is self.project_panel else self.property_drawer
        panel.setProperty("drawerExpanded",expanded)
        if expanded:
            drawer.show(); panel.setFixedWidth(full_width)
        else:
            drawer.hide()
        if action is not None:action.setChecked(expanded)
        self._enforce_splitter_layout()

    def _enforce_splitter_layout(self):
        if not hasattr(self,"main_splitter"):return
        left=300 if self._drawer_expanded(self.project_panel) else max(48,self.project_rail.width())
        right=0
        self.project_panel.setFixedWidth(left); self.property_panel.setFixedWidth(right)
        available=max(240,self.main_splitter.width()-left-right-self.main_splitter.handleWidth()*2)
        self.main_splitter.setSizes([left,available,right])
        if hasattr(self,"web") and self.web_ready:
            QTimer.singleShot(0,self._refresh_cesium_view)

    def _set_scenario_editable(self,editable):
        for widget in self.scenario_edit_widgets:widget.setEnabled(editable)
        if editable:
            self.eccentricity.setEnabled(self.orbit_shape.currentIndex()!=0)
            self._walker_mode_changed(self.walker_mode.currentIndex())
            self._gateway_mode_changed(self.gateway_planning_mode.currentIndex())
        self.property_title.setText("场景参数" if editable else "场景参数 · 仿真期间已锁定")

    def _refresh_scene_preview(self,_value=None):
        if not hasattr(self,"scene_tree"):return
        source="Walker参数" if self.source.currentIndex()==0 else "TLE文件"
        shell_text=f"{self.shell_count.value()}壳层" if self.source.currentIndex()==0 and self.walker_mode.currentIndex()==0 else "单壳层"
        self._set_tree_children(self.scene_tree,[f"轨道来源：{source}",f"Walker方案：{shell_text}",f"卫星：{self.sat_count.value():,} 颗",f"轨道面：{self.planes.value()} 个",f"仿真时长：{self.duration.value():g} s",f"步进：{self.step.value():g} s"])
        stations=[]
        for row in range(self.station_table.rowCount()):
            name=self.station_table.item(row,0); lat=self.station_table.item(row,1); lon=self.station_table.item(row,2)
            if name and lat and lon:stations.append(f"{row+1}. {name.text()}（{lat.text()}°, {lon.text()}°）")
        self._set_tree_children(self.gateway_tree,stations)
        for i in range(self.gateway_tree.childCount()):self.gateway_tree.child(i).setData(0,Qt.ItemDataRole.UserRole,2)
        self.scene_info.setText(f"场景配置预览\n{self.sat_count.value():,}颗卫星 / {self.planes.value()}轨道面\n{len(stations)}个信关站 / {self.duration.value():g}秒")
        self.project_root.setExpanded(True); self.network_layer_tree.setExpanded(True); self.network_scenes_tree.setExpanded(True); self.current_scene_node.setExpanded(True)
        self._refresh_parameter_actions()

    def _refresh_parameter_actions(self,*_values):
        if not getattr(self,"base_parameter_status",None):return
        try:self._config();error=""
        except Exception as exc:error=str(exc)
        if error:
            message=f"● 参数需要完善：{error}"
            color,border,background="#eebfc4","#9d6570","#30252e"
        elif self.worker is not None:
            message="● 基础数据已准备或任务正在运行。需要修改参数时，请先结束当前任务。"
            color,border,background="#d9d3b8","#807b62","#2c2d2b"
        else:
            message="● 基础参数可用：星座、星间链路、信关站、业务和雨衰参数将一起生成基础数据。下一步：点击顶部工具栏的「生成基础数据」。"
            color,border,background="#b9dce8","#367b94","#102c3c"
        for label in self.base_parameter_status:
            label.setText(message)
            label.setStyleSheet(f"QLabel{{color:{color};background:{background};border:1px solid {border};border-radius:5px;padding:8px 12px;font-size:14px;font-weight:700;}}")

    @staticmethod
    def _set_tree_children(parent, labels):
        while parent.childCount(): parent.removeChild(parent.child(0))
        for label in labels: QTreeWidgetItem(parent,[label])

    def _append_priority_region(self,name,lat,lon,radius_km):
        row=self.priority_region_table.rowCount();self.priority_region_table.insertRow(row)
        for column,value in enumerate((name,lat,lon,radius_km)):self.priority_region_table.setItem(row,column,QTableWidgetItem(str(value)))

    def _remove_priority_region(self):
        row=self.priority_region_table.currentRow()
        if row>=0:self.priority_region_table.removeRow(row)

    def _priority_regions(self):
        result=[]
        for row in range(self.priority_region_table.rowCount()):
            try:
                name=self.priority_region_table.item(row,0).text().strip() or f"重点区域{row+1}"
                lat=float(self.priority_region_table.item(row,1).text());lon=float(self.priority_region_table.item(row,2).text());radius=float(self.priority_region_table.item(row,3).text())
                if not -90<=lat<=90 or not -180<=lon<=180 or radius<=0:raise ValueError
                result.append({"name":name,"lat":lat,"lon":lon,"radius_km":radius})
            except (AttributeError,ValueError):raise ValueError(f"重点任务区域第{row+1}行参数无效")
        return result

    def _append_station(self,name,lat,lon,alt=0,antenna_count=20,channel_count=None):
        row=self.station_table.rowCount(); self.station_table.insertRow(row)
        channel_count = antenna_count if channel_count is None else channel_count
        for col,value in enumerate((name,lat,lon,alt,antenna_count,channel_count)):
            item=QTableWidgetItem(str(value))
            if col==5:item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.station_table.setItem(row,col,item)
    def _station_item_changed(self,item):
        if item.column()==4:
            try:value=max(1,int(float(item.text())))
            except ValueError:value=1
            self.station_table.blockSignals(True)
            item.setText(str(value)); self.station_table.item(item.row(),5).setText(str(value))
            self.station_table.blockSignals(False)
        self._refresh_scene_preview()
    def _add_preset_station(self):
        name=self.station_preset.currentText(); self._append_station(name,*PRESET_STATIONS[name]); self._refresh_scene_preview()
    def _remove_station(self):
        rows=sorted({x.row() for x in self.station_table.selectedIndexes()},reverse=True)
        for row in rows:self.station_table.removeRow(row)
        self._refresh_scene_preview()
    def _orbit_shape_changed(self, index):
        circular = index == 0
        if circular:
            self.eccentricity.setValue(0.0000001)
        self.eccentricity.setEnabled(not circular)
    def _orbit_latitude_ceiling(self):
        if self.walker_mode.currentIndex()==0:
            inclinations=[float(shell["inclination_deg"]) for shell in self._walker_shells()]
        else:
            inclinations=[float(self.inclination.value())]
        return min(90.,max((min(value,180.-value) for value in inclinations),default=90.))
    def _latitude_limit_changed(self,_value):
        if not getattr(self,"_setting_lat_limit",False):self._lat_limit_auto=False
    def _sync_default_latitude_limit(self,*_args):
        if not hasattr(self,"lat_limit") or not getattr(self,"_lat_limit_auto",False):return
        try:ceiling=self._orbit_latitude_ceiling()
        except (ValueError,AttributeError):return
        self._setting_lat_limit=True
        try:self.lat_limit.setValue(ceiling)
        finally:self._setting_lat_limit=False
    def _walker_shells(self):
        shells=[]
        for row in range(self.shell_table.rowCount()):
            try:
                name=self.shell_table.item(row,0).text().strip() or f"L{row+1}"
                altitude=float(self.shell_table.item(row,1).text())
                inclination=float(self.shell_table.item(row,2).text())
                planes=int(self.shell_table.item(row,3).text())
                per_plane=int(self.shell_table.item(row,4).text())
            except (AttributeError,ValueError):
                raise ValueError(f"壳层参数表第{row+1}行包含无效参数")
            if altitude<100 or not 0<=inclination<=180 or planes<1 or per_plane<1:
                raise ValueError(f"壳层参数表第{row+1}行参数超出有效范围")
            shells.append({"name":name,"altitude_km":altitude,"inclination_deg":inclination,
                           "planes":planes,"per_plane":per_plane,"satellite_count":planes*per_plane})
        return shells
    def _walker_mode_changed(self,index):
        multi=index==0
        self.shell_count.setVisible(multi);self.shell_table.setVisible(multi)
        for widget in (self.sat_count,self.planes,self.altitude,self.inclination):widget.setEnabled(not multi)
        self._walker_shells_changed();self._sync_default_latitude_limit()
    def _shell_count_changed(self,count):
        old=self.shell_table.rowCount();self.shell_table.blockSignals(True);self.shell_table.setRowCount(count)
        defaults=((508,53,80,100),(508,70,60,100),(508,90,40,100),(508,30,20,100))
        for row in range(old,count):
            altitude,inclination,planes,per_plane=defaults[row] if row<len(defaults) else (508,min(98,30+10*row),20,100)
            for col,value in enumerate((f"L{row+1}",altitude,inclination,planes,per_plane)):
                item=QTableWidgetItem(str(value));item.setFlags(item.flags() if col else item.flags() & ~Qt.ItemFlag.ItemIsEditable);self.shell_table.setItem(row,col,item)
        for row in range(count):
            if self.shell_table.item(row,0):self.shell_table.item(row,0).setText(f"L{row+1}")
        self.shell_table.blockSignals(False);self._walker_shells_changed()
    def _walker_shells_changed(self,_item=None):
        if not hasattr(self,"walker_mode") or self.walker_mode.currentIndex()!=0:return
        try:
            shells=self._walker_shells();total=sum(x["satellite_count"] for x in shells);planes=sum(x["planes"] for x in shells)
        except ValueError:return
        self.sat_count.setValue(total);self.planes.setValue(planes);self.center_plane.setMaximum(planes) if hasattr(self,"center_plane") else None
        if hasattr(self,"station_table"):self._refresh_scene_preview()
        self._sync_default_latitude_limit()
    def _browse_tle(self):
        path,_=QFileDialog.getOpenFileName(self,"选择TLE文件",str(ROOT),"TLE文件 (*.txt *.tle);;所有文件 (*)")
        if path:self.tle_path.setText(path)
    def _browse_traffic_grid(self):
        path,_=QFileDialog.getOpenFileName(self,"选择业务网格CSV",str(ROOT),"业务网格 (*.csv);;所有文件 (*)")
        if path:self.traffic_grid_path.setText(path)
    def _browse_modcod_table(self):
        path,_=QFileDialog.getOpenFileName(self,"选择SNR—速率表",str(ROOT),"CSV表格 (*.csv);;所有文件 (*)")
        if path:self.modcod_table_path.setText(path)
    def _web_loaded(self,ok):
        self.web_ready=ok;self.status.setText("Cesium视图就绪" if ok else "Cesium页面加载失败")
        if ok:QTimer.singleShot(0,lambda:self._planning_subtab_changed(self.planning_tabs.currentIndex()))
    def _stations(self):
        result=[]
        for row in range(self.station_table.rowCount()):
            result.append({"name":self.station_table.item(row,0).text(),"lat":float(self.station_table.item(row,1).text()),"lon":float(self.station_table.item(row,2).text()),"alt":float(self.station_table.item(row,3).text()),"antenna_count":int(float(self.station_table.item(row,4).text())),"channel_count":int(float(self.station_table.item(row,5).text())),"frequency_ghz":self.feeder_frequency.value()})
        return result
    def _gateway_mode_changed(self,index):
        for widget in (self.station_preset,self.add_station_btn,self.station_table,self.remove_station_btn):widget.setEnabled(True)
        self.gateway_planning_enabled.setChecked(False)
    def _update_gsl_distance_hint(self,*_args):
        if not hasattr(self,"gsl_distance_hint"):return
        radius_km=EARTH_RADIUS_M/1000.;altitude_km=max(0.,self.altitude.value());elevation=math.radians(self.elevation.value())
        slant=math.sqrt(max(0.,(radius_km+altitude_km)**2-(radius_km*math.cos(elevation))**2))-radius_km*math.sin(elevation)
        text=f"按主壳层高度 {altitude_km:g} km 计算，最低仰角处参考斜距约 {slant:.1f} km。实际可见性按每颗卫星的位置和高度计算。"
        text+=" 不再使用手工斜距上限。"
        self.gsl_distance_hint.setText(text)

    def _config(self):
        source="walker" if self.source.currentIndex()==0 else "tle"; tle=self.tle_path.text().strip()
        if source=="tle" and not Path(tle).is_file(): raise ValueError("请选择有效的TLE文件")
        speed=float(self.speed.currentText().replace("×",""))
        eccentricity = 0.0000001 if self.orbit_shape.currentIndex() == 0 else self.eccentricity.value()
        display_count = (3, 5, 0)[self.display_planes.currentIndex()]
        isl_pattern = "4isl_grid" if self.isl_pattern.currentIndex() == 0 else "3isl_brick"
        walker_layout = "delta" if self.walker_layout.currentIndex() == 0 else "star"
        seam_enabled = source == "walker" and walker_layout == "star"
        walker_mode="multi_shell" if self.walker_mode.currentIndex()==0 else "single"
        walker_shells=self._walker_shells() if source=="walker" and walker_mode=="multi_shell" else []
        satellite_count=sum(x["satellite_count"] for x in walker_shells) if walker_shells else self.sat_count.value()
        plane_count=sum(x["planes"] for x in walker_shells) if walker_shells else self.planes.value()
        seam_after_plane = plane_count-1
        traffic_path=self.traffic_grid_path.text().strip()
        if traffic_path and not Path(traffic_path).is_file(): raise ValueError("请选择有效的业务网格CSV文件")
        modcod_path=self.modcod_table_path.text().strip()
        if modcod_path and not Path(modcod_path).is_file(): raise ValueError("请选择有效的SNR—速率CSV文件")
        rain_sources=("geographic_auto","manual_sequence","weather_file");rain_sigmas=(0.50,0.35,0.20,0.12)
        stations=self._stations()
        gateway_mode="specified"
        if not stations:raise ValueError("请至少手动添加一个信关站")
        if self.planning_horizon.value()<self.planning_period.value():raise ValueError("规划预测窗口不能短于滚动规划周期")
        if self.planning_slot.value()>self.planning_horizon.value():raise ValueError("规划时间片不能大于规划预测窗口")
        if self.fast_adjust_period.value()<self.fast_evaluation_period.value():raise ValueError("快环调整间隔不能短于状态评估周期")
        capacity_weight=self.planning_capacity_weight.value()/100.0 if self.planning_strategy.currentData()=="window_margin" else 0.0
        delay_weight=1.0-capacity_weight if self.planning_strategy.currentData()=="window_margin" else 0.0
        return SimulationConfig(source,tle,satellite_count,plane_count,walker_layout,self.altitude.value(),self.inclination.value(),eccentricity,self.arg_perigee.value(),self.duration.value(),self.step.value(),speed,self.lat_limit.value(),self.elevation.value(),0.0,self.max_isl.value(),isl_pattern,seam_enabled,seam_after_plane,stations,self.center_plane.value()-1,display_count,self.show_sat.isChecked(),self.show_same.isChecked(),self.show_cross.isChecked(),self.show_gsl.isChecked(),self.show_motion.isChecked(),self.show_labels.isChecked(),self.traffic_enabled.isChecked(),traffic_path,self.traffic_dense_percentile.value(),self.traffic_dense_max_satellites.value(),self.traffic_random_sigma.value(),self.traffic_outside_china_percent.value(),self.traffic_total_gbps.value(),self.traffic_cluster_radius.value(),self.planning_enabled.isChecked(),self.planning_period.value(),self.feeder_bandwidth.value(),self.snr_margin.value(),modcod_path,85.0,self.switching_loss.value(),self.rain_enabled.isChecked(),rain_sources[self.rain_source.currentIndex()],rain_sigmas[self.rain_sense.currentIndex()],self.rain_forecast.value(),self.rain_p90.isChecked(),self.satellite_antennas.value(),self.satellite_channels.value(),self.satellite_channel_capacity.value(),self.geo_interference.isChecked(),self.gateway_eirp.value(),self.gateway_gt.value(),self.gateway_array_loss.value(),walker_mode,walker_shells,"","",self.gateway_planning_enabled.isChecked(),5,self.gateway_max_sites.value(),self.gateway_redundancy.value(),self.gateway_max_antennas.value(),gateway_mode,self.planning_horizon.value(),self.planning_slot.value(),self.fast_evaluation_period.value(),self.fast_adjust_period.value(),self.route_update_period.value(),30.0,self.traffic_landing_percent.value(),self.geo_satellite_count.value(),self.geo_center_longitude.value(),self.geo_longitude_spacing.value(),self.geo_altitude.value(),self.gso_i_over_n.value(),self.gso_protection_angle.value(),self.planning_capacity_reserve.value(),self._priority_regions(),capacity_weight,delay_weight,self.feeder_frequency.value(),self.feeder_rolloff.value(),self.satellite_eirp.value(),self.planning_strategy.currentData())
    @Slot()
    def _apply_scheduling_parameters(self):
        if self.fast_adjust_period.value()<self.fast_evaluation_period.value():
            QMessageBox.warning(self,"调度参数","最短调整间隔不能短于状态评估周期。")
            return
        if self.worker is not None:
            self.worker.set_scheduling_parameters(self.fast_evaluation_period.value(),self.fast_adjust_period.value(),self.route_update_period.value(),30.0,85.0/100.0,self.switching_loss.value()/100.0)
            self.status.setText("调度参数已应用，从下一仿真帧生效；不会重新执行离线规划")
        else:self.status.setText("调度参数已保存，开始仿真后生效")

    @Slot()
    def _run_all_stages(self):
        """Run preparation, offline planning and simulation as one workflow."""
        if self.loaded_result:
            QMessageBox.information(self,"一键运行","请先关闭当前仿真结果回放。")
            return
        if self.worker is not None and self.prepared_ready and self.is_paused and self._simulation_at_end():
            self.stop_simulation()
            self._run_all_after_stop=True
            return
        if self.worker is not None and not (self.prepared_ready and self.is_paused and self.current_full_frame is None):
            QMessageBox.information(self,"一键运行","当前任务正在运行，请结束后再执行一键运行。")
            return
        if not self.web_ready:
            QMessageBox.warning(self,"视图未就绪","请等待三维地图加载完成。")
            return
        try:cfg=self._config()
        except Exception as exc:
            QMessageBox.warning(self,"参数错误",str(exc));return
        if abs(float(cfg.duration_s)-1800.0)>1e-9:
            QMessageBox.warning(self,"仿真时长","离线规划固定覆盖30分钟，请将仿真时长设置为1800秒。")
            return
        self._one_click_running=True;self.run_all_btn.setText("一键运行中…");self._sync_run_controls()
        if self.worker is None:
            self.status.setText("一键运行：生成或加载轨道与业务基础数据；随后自动进入离线规划")
            self.start_simulation()
        else:
            self.status.setText("一键运行：生成或加载18个离线规划方案；随后自动开始仿真")
            QTimer.singleShot(0,self._run_offline_planning)

    def _reset_one_click_run(self):
        self._one_click_running=False
        if hasattr(self,"run_all_btn"):
            self.run_all_btn.setText("一键运行")
            self._sync_run_controls()

    def _simulation_at_end(self):
        for frame in (self.latest_full_frame,self.current_full_frame):
            if frame and int(frame.get("total",0))>0 and int(frame.get("index",-1))>=int(frame["total"])-1:
                return True
        return False

    def _sync_run_controls(self):
        """Keep workflow actions consistent with the worker and planning state."""
        if not hasattr(self,"run_all_btn"):return
        if self.loaded_result:
            self.run_all_btn.setEnabled(False);self.prepare_btn.setEnabled(False);self.offline_plan_btn.setEnabled(False)
            self.pause_btn.setEnabled(self.result_playing);self.stop_btn.setEnabled(True)
            return
        active=self.worker is not None
        ready=active and self.prepared_ready and self.is_paused
        planning=self.offline_planning_started_at is not None
        busy=self.user_stopping or self.regenerate_pending or planning
        idle=not active and self.thread is None
        initial=ready and self.current_full_frame is None
        completed=ready and self._simulation_at_end()
        self.run_all_btn.setEnabled(not self._one_click_running and not busy and (idle or initial or completed))
        self.prepare_btn.setEnabled(not self._one_click_running and not busy and (idle or initial))
        self.offline_plan_btn.setEnabled(not self._one_click_running and not busy and ready)
        self.start_btn.setEnabled(not busy and ready and self.offline_rain_grade.currentData() is not None)
        self.pause_btn.setEnabled(not busy and active and self.prepared_ready and not self.is_paused)
        self.stop_btn.setEnabled(active and not self.user_stopping)

    @Slot()
    def _run_offline_planning(self):
        if self.worker is None or not self.prepared_ready:
            QMessageBox.information(self,"离线规划","请先生成30分钟场景的基础数据。")
            return
        self.offline_plan_btn.setEnabled(False);self.start_btn.setEnabled(False);self.offline_rain_grade.setEnabled(False)
        self.state_badge.setText("● 离线规划中");self.status.setText("正在生成轻雨、中雨、重雨三种工况的18个离线方案…")
        self.offline_planning_started_at=time.perf_counter();self.busy_overlay.start("正在检查规划参数与基础数据缓存…")
        self._sync_run_controls()
        strategy=self.planning_strategy.currentData()
        capacity=self.planning_capacity_weight.value()/100.0 if strategy=="window_margin" else 0.0
        delay=1.0-capacity if strategy=="window_margin" else 0.0
        if self.active_config is not None:
            self.active_config.planning_capacity_weight=capacity
            self.active_config.planning_delay_weight=delay
            self.active_config.planning_capacity_reserve_percent=self.planning_capacity_reserve.value()
            self.active_config.planning_selection_strategy=strategy
        self.worker.request_offline_planning(capacity,delay,self.planning_capacity_reserve.value()/100.0,strategy)

    def _populate_planning_choices(self,catalog):
        previous=self.offline_rain_grade.currentData() or "轻雨"
        groups={grade:[row for row in catalog if row.get("rain_grade")==grade] for grade in ("轻雨","中雨","重雨")}
        self.offline_rain_grade.blockSignals(True);self.offline_rain_grade.clear()
        for grade,plans in groups.items():
            if len(plans)!=6:continue
            strategy="容量时延加权" if plans[0].get("selection_strategy")=="window_margin" else "容量可行优先时延"
            index=self.offline_rain_grade.count();self.offline_rain_grade.addItem(f"{grade} · 6个时段 · {strategy}",grade)
            self.offline_rain_grade.setItemData(index,f"本次规划批次：{str(plans[0].get('planning_spec_hash','--'))[:12]}",Qt.ItemDataRole.ToolTipRole)
        if not self.offline_rain_grade.count():self.offline_rain_grade.addItem("尚无完整可执行的规划方案",None)
        index=self.offline_rain_grade.findData(previous)
        if index<0:index=self.offline_rain_grade.findData("轻雨")
        self.offline_rain_grade.setCurrentIndex(max(0,index));self.offline_rain_grade.setEnabled(self.offline_rain_grade.currentData() is not None)
        self.offline_rain_grade.blockSignals(False)

    @Slot(list)
    def _offline_planning_ready(self,catalog):
        if self.user_stopping:return
        self.busy_overlay.stop();self.offline_planning_started_at=None
        self.planning_results.set_offline_catalog(catalog);self.planning_results.result_tabs.setCurrentIndex(0);self.planning_tabs.setCurrentIndex(1);self.offline_plan_btn.setEnabled(True)
        self._populate_planning_choices(catalog)
        if self.offline_rain_grade.currentData() is None:
            self._reset_one_click_run()
            self.start_btn.setEnabled(False);self.state_badge.setText("● 离线规划失败");self.status.setText("离线规划未生成任何方案，请检查规划日志后重试")
            QMessageBox.critical(self,"离线规划失败","没有生成可执行的离线方案，仿真无法启动。")
            return
        infeasible=[row for row in catalog if not row.get("delivery_target_satisfied",False)]
        # An infeasible plan is still a valid simulation input: running it is
        # how users inspect the shortfall and scheduling behaviour.  Only an
        # empty catalog blocks simulation.
        self.start_btn.setEnabled(True)
        if infeasible:
            worst=max(infeasible,key=lambda row:float(row.get("delivery_gap_gbps",0)))
            self.state_badge.setText("● 离线方案有容量缺口")
            self.status.setText(f"离线规划已生成，可开始仿真；{len(infeasible)}/{len(catalog)} 个方案未达到99%，最大缺口 {float(worst.get('delivery_gap_gbps',0)):.3f} Gb/s，详见“规划方案→规划批次”")
            if not self._one_click_running:
                QMessageBox.warning(self,"离线规划完成但存在缺口",f"已生成 {len(catalog)} 个方案，可以开始仿真。\n\n其中 {len(infeasible)} 个方案未达到99%计划送达目标，最大缺口 {float(worst.get('delivery_gap_gbps',0)):.3f} Gb/s。\n已自动打开“规划批次”，可查看每个时段和雨衰工况的容量校核。")
        else:
            self.state_badge.setText("● 离线方案就绪");self.status.setText(f"18个方案均达到99%送达目标；已选择 {self.offline_rain_grade.currentText()}")
        self.worker.set_offline_rain_grade(self.offline_rain_grade.currentData())
        self._sync_run_controls()
        if self._one_click_running:
            self.status.setText(f"一键运行：开始仿真，执行 {self.offline_rain_grade.currentText()}")
            QTimer.singleShot(0,self._primary_action)

    @Slot(int)
    def _select_offline_rain_grade(self,_index):
        grade=self.offline_rain_grade.currentData()
        if grade is None:return
        if self.worker is not None:self.worker.set_offline_rain_grade(grade)
        self.status.setText(f"实时调度将采用“{grade}”离线规划方案，不执行紧急规划")

    @Slot()
    def start_simulation(self):
        if self.worker is not None:
            if self.prepared_ready and self.is_paused and self.current_full_frame is None:
                self.regenerate_pending=True;self.prepare_btn.setEnabled(False);self.start_btn.setEnabled(False)
                self.state_badge.setText("● 正在重新准备");self.status.setText("正在释放旧的基础数据…")
                self._sync_run_controls()
                self.worker.stop()
                return
            QMessageBox.information(self,"仿真已经开始","请先结束当前仿真，再重新生成基础数据。")
            return
        if not self.web_ready: QMessageBox.warning(self,"视图未就绪","请等待Cesium页面加载完成"); return
        try: cfg=self._config()
        except Exception as exc: QMessageBox.warning(self,"参数错误",str(exc)); return
        if abs(float(cfg.duration_s)-1800.0)>1e-9:
            QMessageBox.warning(self,"仿真时长","离线规划固定覆盖30分钟，请将仿真时长设置为1800秒后重新生成基础数据。")
            return
        if self.result_store:self.result_store.close()
        _scenario_id,_scenario_hash,scenario_dir=self._public_scenario(cfg)
        cfg.scenario_directory=str(scenario_dir);cfg.result_directory=""
        self.result_store=None;self.result_store_path=None;self.requested_result_path=None;self.loaded_result=False
        self.active_config=cfg; self.prepared_ready=False; self.prepared_metadata=None; self.summary_history.clear(); self.planning_results.clear();self.planning_resources.clear();self.planning_evaluation.clear();self._populate_planning_choices([]); self._refresh_parameter_actions(); self.preparation_started_at=time.perf_counter();self.run_started_at=None;self.run_elapsed_s=0.0;self.elapsed_phase="preparation";self.elapsed_label.setText("准备耗时 00:00:00");self.cache_status_label.setText("缓存检查中");self.cache_status_label.setToolTip("正在检查各模型缓存");self.cache_status_label.setStyleSheet("QLabel{color:#ff6675;background:#35131b;border:1px solid #ff3b4f;border-radius:3px;padding:3px 8px;font-weight:700}"); self.state_badge.setText("● 准备中"); self.alert_label.setText("告警 0"); self._set_scenario_editable(False);self.busy_overlay.start("正在检查并生成场景基础数据…")
        self._set_property_drawer(False)
        self.user_stopping = False; self.prepared_ready = False
        self.latest_full_frame = None; self.current_full_frame = None; self.scenario_planes = None; self.pending_frame = None
        self.last_transfer_page = None; self.last_sent_isls = {}
        self.frame_cache.clear(); self.follow_live = True; self.is_paused = True
        self.prepare_btn.setText("正在生成基础数据…");self.prepare_btn.setEnabled(False);self.start_btn.setText("开始仿真"); self.start_btn.setEnabled(False); self.timeline.setEnabled(False)
        self.compute_progress.setValue(0); self.transfer_progress.setValue(0); self.stop_btn.setEnabled(True)
        self.thread=QThread(self); self.worker=SimulationWorker(cfg,self.result_store);self.worker.set_display_options(self._filter_options())
        
        self.worker.moveToThread(self.thread); self.thread.started.connect(self.worker.run); self.worker.prepared.connect(self._prepared); self.worker.offline_planning_ready.connect(self._offline_planning_ready); self.worker.display_frame_ready.connect(self._queue_frame); self.worker.frame_ready.connect(self._update_planning_views); self.worker.progress.connect(self._progress); self.worker.paused_changed.connect(self._worker_paused); self.worker.failed.connect(self._failed); self.worker.finished.connect(self._finished); self.worker.finished.connect(self.thread.quit); self.thread.finished.connect(self._thread_stopped); self.thread.finished.connect(self.thread.deleteLater); self.thread.start()
        self._sync_run_controls()
    @Slot(dict)
    def _prepared(self,data):
        if self.user_stopping:return
        self.busy_overlay.stop()
        preparation_seconds=max(0,int(time.perf_counter()-self.preparation_started_at)) if self.preparation_started_at is not None else 0
        data["preparation_elapsed_s"]=preparation_seconds
        self.elapsed_phase="prepared";self.preparation_started_at=None;self.elapsed_label.setText(f"准备耗时 {preparation_seconds//3600:02d}:{preparation_seconds//60%60:02d}:{preparation_seconds%60:02d}")
        if self.result_store and not self.loaded_result:
            scenario_reference=self.result_store.get_metadata("scenario_reference",{})
            data["scenario_id"]=scenario_reference.get("scenario_id","");data["scenario_reference"]="scenario_reference.json"
            data["scenario_orbit_path"]="base_data/orbit_positions.npy";data["scenario_traffic_path"]="base_data/traffic_timeseries.sqlite"
        if self.result_store and not self.loaded_result:
            self.result_store.set_metadata("prepared",data)
            self.result_store.set_metadata("config",asdict(self.active_config))
        self.prepared_metadata=dict(data)
        self.scenario_planes = data["planes"]
        data["layers"] = self._layer_options()
        data["priority_regions"] = list(self.active_config.priority_task_regions)
        self.frame_cache_limit = max(5, min(120, 500_000 // max(1, data["satellite_count"])))
        self.timeline.setRange(0, max(0, data["frames"]-1)); self.timeline.setValue(0); self.timeline.setEnabled(True)
        self.web.page().runJavaScript(f"configureScenario({json.dumps(data,ensure_ascii=False)})")
        shell_summary="；".join(f"{x.get('name','壳层')} {int(x['planes'])}面×{int(x['per_plane'])}星" for x in data.get("walker_shells",[]))
        shell_summary=shell_summary or f"每轨{data['per_plane']}颗"
        self._set_tree_children(self.scene_tree,["状态：已生成",f"轨道对象：{data['satellite_count']:,}颗",f"轨道面：{data['planes']}个",f"壳层配置：{shell_summary}",f"时间轴：{data['frames']}点 / 步进{data['step_s']:g}s",f"历元：{data['epoch']}"])
        self._set_tree_children(self.isl_tree,["状态：候选关系已生成",f"ISL候选：{data.get('isl_candidate_count',0):,}条",f"缝隙候选：{data.get('seam_candidate_count',0):,}条",f"构型：{self.isl_pattern.currentText()}",f"纬度门限：±{self.lat_limit.value():g}°"])
        self._set_tree_children(self.gateway_tree,[f"{i+1}. {gs['name']}（{gs['lat']:.4f}°, {gs['lon']:.4f}°）" for i,gs in enumerate(data['stations'])])
        for i in range(self.gateway_tree.childCount()):self.gateway_tree.child(i).setData(0,Qt.ItemDataRole.UserRole,2)
        self._set_tree_children(self.traffic_tree,["状态：空间权重已载入",f"业务网格：{data.get('traffic_cell_count',0):,}个",f"业务基准：{self.traffic_total_gbps.value():g} Gb/s",f"业务落地比例：{self.traffic_landing_percent.value():g}%",f"规划采样：{data.get('traffic_keyframe_count',0)}个 / {data.get('traffic_update_period_s',30):g}s",f"时间演化：地方时双峰曲线"])
        self._set_tree_children(self.rain_tree,["状态：参数已配置",f"来源：{self.rain_source.currentText()}",f"感知：{self.rain_sense.currentText()}",f"预报提前期：{self.rain_forecast.value():g} h",f"规划上界：{'P90' if self.rain_p90.isChecked() else '均值'}","统计工况：轻雨5% / 中雨1% / 重雨0.1%超越概率"])
        self._set_tree_children(self.results_tree,["当前帧：等待播放","可见卫星：--","有效ISL：--","GSL：--"])
        cache_text=f"已缓存 {data.get('orbit_cache_size_mb',0):g} MB" if data.get("orbit_cache") else "按需计算";traffic_cache_text=("关闭" if not self.active_config.traffic_enabled else "已复用" if data.get("traffic_cache_reused") else "已生成" if data.get("traffic_cache") else "按需计算")
        deployment=data.get("gateway_deployment") or {}
        orbit_text=("复用" if data.get("orbit_cache_reused") else "新算") if data.get("orbit_cache") else "按需计算"
        topology_text="未变" if data.get("isl_model_reused") else "已更新"
        gateway_text=("复用" if data.get("gateway_cache_reused") else "新算") if deployment.get("mode")=="automatic" else "手动"
        rain_text=("未变" if data.get("rain_cache_reused") else "已更新") if self.active_config.rain_enabled else "关闭"
        cache_states=[("轨道",orbit_text),("ISL模型",topology_text),("业务时变",traffic_cache_text),("信关站",gateway_text),("雨衰参数",rain_text)]
        reused=sum(value in ("复用","已复用","未变") for _,value in cache_states)
        updated=sum(value in ("新算","已生成","已更新") for _,value in cache_states)
        self.cache_status_label.setText(f"缓存 {reused}复用·{updated}更新")
        self.cache_status_label.setToolTip("\n".join(f"{name}：{value}" for name,value in cache_states))
        self.cache_status_label.setStyleSheet("QLabel{color:#72ffb0;background:#0b3324;border:1px solid #20df83;border-radius:3px;padding:3px 8px;font-weight:700}")
        self.planning_results.set_gateway_deployment(deployment)
        if deployment.get("mode")=="automatic":
            self.station_table.blockSignals(True);self.station_table.setRowCount(0)
            self.station_table.setHorizontalHeaderLabels(["自动规划结果","纬度","经度","高度m","规划天线","规划信道"])
            for station in data.get("stations",[]):self._append_station(station.get("name","自动站"),station.get("lat",0),station.get("lon",0),station.get("alt",0),station.get("antenna_count",1),station.get("channel_count",1))
            self.station_table.blockSignals(False)
        else:self.station_table.setHorizontalHeaderLabels(["手动设置信关站","纬度","经度","高度m","天线数","信道数"])
        gateway_mode_text="手动设置"
        self.scene_info.setText(f"五类模型基础数据就绪\n星座：{data['satellite_count']:,}颗 / {data['planes']}个轨道面\n轨道位置：{cache_text}\n业务时变数据：{traffic_cache_text}\n星间链路：{data.get('isl_candidate_count',0):,}条候选\n信关站：{gateway_mode_text} / {len(data['stations'])}站 / 单信道2.0 Gb/s\n业务：{data.get('traffic_cell_count',0):,}个网格\n雨衰：{self.rain_source.currentText()} / 预报{self.rain_forecast.value():g}h\n时间轴：{data['frames']:,}点，步进{data['step_s']:g}s")
        self.current_scene_node.setText(0,"当前场景（已准备）"); self.state_badge.setText("● 已准备")
        self.project_root.setExpanded(True); self.network_layer_tree.setExpanded(True); self.network_scenes_tree.setExpanded(True); self.current_scene_node.setExpanded(True)
        self.prepared_ready = True; self.is_paused = True; self.start_btn.setText("开始仿真"); self.start_btn.setEnabled(False);self.offline_plan_btn.setEnabled(True);self._refresh_parameter_actions()
        self.worker.set_paused(True);self.prepare_btn.setText("重新生成基础数据");self.prepare_btn.setEnabled(True)
        self._sync_run_controls()
        self.status.setText(f"基础数据已建立：已载入 {len(data['stations'])} 个手动设置信关站，单信道能力2.0 Gb/s。")
        self.state_badge.setText("● 场景就绪")
        self._show_notice(
            f"✓ 五类模型基础数据已就绪　星座 {data['satellite_count']:,}星　星间链路 {data.get('isl_candidate_count',0):,}候选　"
            f"手动设置信关站 {len(data['stations'])}站　单信道2.0 Gb/s　业务 {data.get('traffic_cell_count',0):,}网格　雨衰参数已配置")
        if self._one_click_running:
            self.status.setText("一键运行：生成或加载离线规划")
            QTimer.singleShot(0,self._run_offline_planning)
    @Slot(dict)
    def _queue_frame(self,data):
        # "Simulation elapsed" measures the displayed t=0 -> t=end interval.
        # Initial global planning belongs to preparation/initialization and must
        # not add several seconds before simulation time has started.
        if self._simulation_clock_pending and int(data.get("index",-1))==0:
            self.run_started_at=time.perf_counter();self.run_elapsed_s=0.0
            self.elapsed_phase="simulation";self._simulation_clock_pending=False
            self.elapsed_label.setText("仿真耗时 00:00:00")
        # The simulated interval ends when the final frame has been computed
        # and delivered to Qt.  Cesium may still draw that frame afterwards;
        # map rendering is presentation time and must not inflate simulation
        # elapsed time.
        if int(data.get("index",-1))==int(data.get("total",0))-1 and self.run_started_at is not None:
            self.run_elapsed_s+=max(0.0,time.perf_counter()-self.run_started_at);self.run_started_at=None
            seconds=max(0,int(self.run_elapsed_s));self.elapsed_label.setText(f"仿真耗时 {seconds//3600:02d}:{seconds//60%60:02d}:{seconds%60:02d}")
        emitted_perf=data.pop("_emitted_perf",None)
        if emitted_perf is not None and self.active_config:
            append_diagnostic(str(Path(self.active_config.result_directory)/"time_variation.jsonl") if self.active_config.result_directory else "","frame_received",frame=data["index"],qt_queue_ms=round((time.perf_counter()-emitted_perf)*1000,1),pending_render_replaced=self.pending_frame is not None,transfer_in_flight=self.transfer_in_flight)
        self.latest_full_frame = data
        if self.is_paused and self._simulation_at_end():
            if self._one_click_running:self._reset_one_click_run()
            else:self._sync_run_controls()
        summary=data.get("summary")
        if summary is not None:self.summary_history[data["index"]]=summary
        if self.seeking_index is not None:
            if data["index"]!=self.seeking_index:return
            self.seeking_index=None;self.follow_live=True;self._show_full_frame(data);return
        if not self.follow_live:
            return
        speed=float(self.speed.currentText().replace("×",""));now=time.perf_counter();is_final=int(data.get("index",-1))==int(data.get("total",0))-1
        # Cesium needs about 0.4 s for a planning-map frame on the 20,000-star
        # scenario.  Sending every 1 s simulation frame can build a GUI queue
        # even though the numerical worker remains within budget.  Two map
        # refreshes per second stay smooth and always retain the newest frame.
        display_interval=0.50 if speed<2 else (0.35 if speed<5 else 0.50)
        if not is_final and display_interval and now-self.last_display_at<display_interval:return
        self.last_display_at=now
        self.frame_cache[data["index"]]=data;self.frame_cache.move_to_end(data["index"])
        while len(self.frame_cache)>self.frame_cache_limit:self.frame_cache.popitem(last=False)
        self._show_full_frame(data)

    @Slot(dict)
    def _update_planning_views(self,data):
        """Update Qt result pages from the complete frame, independently of map filtering."""
        if not data or not data.get("planning") or not self.active_config:return
        self.current_full_frame=data
        station_names=[x["name"] for x in self.active_config.stations]
        for selector in (self.planning_results.gateway_scope,self.planning_resources.gateway_scope,self.planning_evaluation.gateway_scope):
            _sync_gateway_scope(selector,station_names)
            selected=selector.findData(self.selected_result_gateway)
            if selected>=0 and selected!=selector.currentIndex():selector.setCurrentIndex(selected)
        self.planning_results.update_frame(data,station_names)
        self.planning_resources.update_frame(data,station_names)
        self.planning_evaluation.update_frame(data,station_names)
        if data["planning"].get("planning_event"):
            self.status.setText(f"慢环第 {data['planning'].get('generation',0)} 次规划已到达；快环正在按当前链路状态调度")

    def _show_full_frame(self, data):
        display_filtered=bool(data.get("_display_filtered",False))
        if display_filtered:
            data.pop("_display_filtered",None);filter_ms=data.pop("_render_filter_ms",0)
            append_diagnostic(str(Path(self.active_config.result_directory)/"time_variation.jsonl") if self.active_config and self.active_config.result_directory else "","frame_filtered",frame=data.get("index",-1),page=self._filter_options()["page"],filter_ms=filter_ms,filter_queue_ms=0,rows={name:len(data.get(name,[])) for name in ("satellites","isls","seams","gsls","traffic","satellite_traffic")})
            self.pending_frame=data
            if not self.transfer_queue and not self.transfer_in_flight:self._start_transfer()
            return
        self.current_full_frame = data
        absolute_time = data.get("absolute_time", "--").replace("+00:00", "Z")
        summary=data.get("summary") or {};visible_satellites=int(summary.get("visible_satellite_count",len({row[1] for row in data["gsls"]})));same_count=int(summary.get("same_isl_count",0));cross_count=int(summary.get("cross_isl_count",len(data["isls"])-same_count))
        planning=data.get("planning") or {}; active=planning.get("active_evaluation") or {}
        labels=[f"当前帧：{data['index']+1}/{data['total']}",f"可见卫星：{visible_satellites}",f"同轨ISL：{same_count}",f"异轨ISL：{cross_count}",f"GSL：{len(data['gsls'])}",f"业务网格：{len(data.get('traffic',[]))}",f"承载业务卫星：{len(data.get('satellite_traffic',[]))}",f"慢环规划批次：{planning.get('generation',0)}",f"快环调度连接：{active.get('active_link_count',0)}",f"调度动作：{planning.get('action','--')}"]
        self._set_tree_children(self.results_tree,labels)
        self.scene_info.setText(f"UTC {absolute_time}\nt={data['time_s']:g} s · 可见卫星 {visible_satellites}\nISL {len(data['isls']):,} · GSL {len(data['gsls']):,}")
        self.results_tree.setExpanded(True)
        if not display_filtered:self._update_planning_views(data)
        # Table and chart pages do not need a hidden Cesium frame. Returning to
        # either map page requests the latest full frame through the worker.
        if hasattr(self,"planning_tabs") and self.planning_tabs.currentIndex() in (1,2,3):
            self.pending_frame=None;self.transfer_queue.clear();self.transfer_progress.setValue(0);return
        if data.pop("_display_filtered",False):
            filter_ms=data.pop("_render_filter_ms",0)
            append_diagnostic(str(Path(self.active_config.result_directory)/"time_variation.jsonl") if self.active_config and self.active_config.result_directory else "","frame_filtered",frame=data.get("index",-1),page=self._filter_options()["page"],filter_ms=filter_ms,filter_queue_ms=0,rows={name:len(data.get(name,[])) for name in ("satellites","isls","seams","gsls","traffic","satellite_traffic")})
            self.pending_frame=data
            if not self.transfer_queue and not self.transfer_in_flight:self._start_transfer()
        else:self._schedule_frame_filter(data)
    def _timeline_pressed(self):
        self.timeline_dragging=True;self.follow_live=False;self.seeking_index=None
        self.timeline_resume_after_drag=self.result_playing if self.loaded_result else bool(self.worker is not None and not self.is_paused)
        self.pending_frame=None;self.transfer_queue.clear();self.transfer_progress.setValue(0)
        if self.loaded_result and self.result_playing:self._toggle_result_playback()
        elif self.worker is not None and not self.is_paused:self._toggle_pause()
        if self.web_ready:self.web.page().runJavaScript("setTimelineDragging(true)")
        self.status.setText("拖动时间轴：仅预览时刻，松开后定位并恢复原播放状态")
    def _timeline_released(self):
        self.timeline_dragging=False
        if self.web_ready:self.web.page().runJavaScript("setTimelineDragging(false)")
        index = self.timeline.value()
        if self.loaded_result:
            self.follow_live=True
            resume=self.timeline_resume_after_drag;self.timeline_resume_after_drag=False
            self.state_badge.setText("● 正在读取");self.status.setText(f"正在从结果包读取第 {index} 帧…")
            def load_and_resume():
                self._load_result_frame(index)
                if resume and not self.result_playing:self._toggle_result_playback()
            QTimer.singleShot(0,load_and_resume);return
        frame = self.frame_cache.get(index)
        if frame is not None:
            self.follow_live=True;self._show_full_frame(frame);self.status.setText(f"已定位到 t={frame['time_s']:g} s")
            if self.worker is not None:self.seeking_index=index;self.worker.seek(index)
        elif self.worker is not None:
            self.seeking_index=index;self.worker.seek(index);self.status.setText(f"正在定位到 t={index*self.step.value():g} s…")
        if self.timeline_resume_after_drag and self.worker is not None:
            self.is_paused=False;self.run_started_at=time.perf_counter();self.elapsed_phase="simulation";self.worker.set_paused(False);self.start_btn.setText("仿真运行中");self.start_btn.setEnabled(False);self.prepare_btn.setEnabled(False);self.state_badge.setText("● 运行中")
        self.timeline_resume_after_drag=False
    def _timeline_moved(self, index):
        frame = self.frame_cache.get(index)
        time_s = frame["time_s"] if frame is not None else index * self.step.value()
        self.timeline_label.setText(f"t={time_s:g} s")
    def _primary_action(self):
        if self.loaded_result:self._toggle_result_playback(); return
        if self.worker is None:
            QMessageBox.information(self,"请先生成基础数据","先点击工具栏中的“生成基础数据”，完成离线数据准备后再开始仿真。")
            return
        if not self.prepared_ready:
            return
        if self.is_paused:
            restarting=self._simulation_at_end()
            if restarting:self.worker.seek(0)
            self.is_paused=False
            if restarting or self.current_full_frame is None or self.timeline.value()==0:
                self.run_elapsed_s=0.0;self.elapsed_phase="simulation";self.elapsed_label.setText("仿真耗时 00:00:00")
            self.run_started_at=None;self._simulation_clock_pending=True
            self.worker.set_paused(False);self.start_btn.setText("仿真运行中");self.start_btn.setEnabled(False);self.prepare_btn.setEnabled(False);self._refresh_parameter_actions()
            self.status.setText("时变推演运行中：正在逐步计算拓扑、时空业务和规划调度")
            self.state_badge.setText("● 运行中")
            self._sync_run_controls()
    def _toggle_pause(self):
        if self.worker is None: return
        if self.run_started_at is not None:
            self.run_elapsed_s+=max(0.0,time.perf_counter()-self.run_started_at);self.run_started_at=None
        self.is_paused=True;self.worker.set_paused(True);self.start_btn.setText("继续仿真");self.start_btn.setEnabled(True);self._refresh_parameter_actions()
        self.status.setText("时变推演已暂停，可查看结果或拖动时间轴");self.state_badge.setText("● 已暂停")
        self._sync_run_controls()
    @Slot(bool)
    def _worker_paused(self, paused):
        self.is_paused = paused
        at_end = self._simulation_at_end()
        if paused and self.run_started_at is not None:
            self.run_elapsed_s+=max(0.0,time.perf_counter()-self.run_started_at);self.run_started_at=None
            seconds=max(0,int(self.run_elapsed_s));self.elapsed_label.setText(f"仿真耗时 {seconds//3600:02d}:{seconds//60%60:02d}:{seconds%60:02d}")
        self.start_btn.setText("重新仿真" if paused and at_end else ("继续仿真" if paused else "仿真运行中"))
        self.start_btn.setEnabled(paused)
        self.state_badge.setText("● 已完成" if paused and at_end else ("● 已暂停" if paused else "● 运行中"))
        if paused and at_end and self._one_click_running:self._reset_one_click_run()
        else:self._sync_run_controls()
    def _speed_changed(self, text):
        if self.worker is not None: self.worker.set_speed(float(text.replace("×","")))
        if self.loaded_result and self.result_playing:self.result_play_timer.start(self._result_interval_ms())
    def _filter_options(self):
        planes=self.scenario_planes or self.planes.value()
        planning_page=self.planning_tabs.currentIndex() if hasattr(self,"planning_tabs") else 0
        page=(0 if self.input_view_mode=="topology" else 1) if planning_page==4 else 2
        return {"planes":planes,"count":(3,5,0)[self.display_planes.currentIndex()],"center":(self.center_plane.value()-1)%planes,"page":page,"subpage":planning_page,"bounds":self.map_view_bounds,"station_count":len(self.active_config.stations) if self.active_config else 0}

    def _sync_worker_display_options(self):
        if self.worker is not None:
            self.worker.set_display_options(self._filter_options())
            full_frame=self.worker.latest_full_frame()
            if full_frame is not None:self._schedule_frame_filter(full_frame)

    def _schedule_frame_filter(self,frame):
        self.render_filter_generation+=1;generation=self.render_filter_generation;options=self._filter_options()
        if self.render_filter_future is not None and not self.render_filter_future.done():self.render_filter_future.cancel()
        submitted_at=time.perf_counter()
        def filter_job():
            started=time.perf_counter();filtered=self._filter_frame(frame,options)
            filtered["_render_filter_ms"]=round((time.perf_counter()-started)*1000,1);filtered["_render_filter_queue_ms"]=round((started-submitted_at)*1000,1)
            return filtered
        future=self.render_filter_executor.submit(filter_job);self.render_filter_future=future
        def completed(item):
            if item.cancelled():return
            try:self.filtered_frame_ready.emit(item.result(),generation)
            except Exception as exc:append_diagnostic(str(Path(self.active_config.result_directory)/"time_variation.jsonl") if self.active_config and self.active_config.result_directory else "","render_filter_error",message=f"{type(exc).__name__}: {exc}")
        future.add_done_callback(completed)

    @Slot(dict,int)
    def _filtered_frame_ready(self,frame,generation):
        if self._closing or generation!=self.render_filter_generation:return
        append_diagnostic(str(Path(self.active_config.result_directory)/"time_variation.jsonl") if self.active_config and self.active_config.result_directory else "","frame_filtered",frame=frame.get("index",-1),page=self._filter_options()["page"],filter_ms=frame.pop("_render_filter_ms",0),filter_queue_ms=frame.pop("_render_filter_queue_ms",0),rows={name:len(frame.get(name,[])) for name in ("satellites","isls","seams","gsls","traffic","satellite_traffic")})
        self.pending_frame=frame
        if not self.transfer_queue and not self.transfer_in_flight:self._start_transfer()

    @staticmethod
    def _filter_frame(frame, options):
        planes=options["planes"]
        count=options["count"]
        center=options["center"]
        if count == 0 or count >= planes:
            visible_planes = set(range(planes))
        else:
            radius = count // 2
            visible_planes = {(center + offset) % planes for offset in range(-radius, radius + 1)}
        visible_satellites = {row[0] for row in frame["satellites"] if row[1] in visible_planes}
        page=options["page"]
        detail_view=False
        map_bounds=options["bounds"]
        if map_bounds:
            west,south,east,north=map_bounds;lon_span=(east-west)%360;lat_span=north-south
            if lon_span<120 and lat_span<90:
                detail_view=True
                def in_view(row):
                    lat,lon=row[2],row[3];longitude_ok=west<=lon<=east if west<=east else lon>=west or lon<=east
                    return south<=lat<=north and longitude_ok
                visible_satellites&={row[0] for row in frame["satellites"] if in_view(row)}
        if page==0 and not detail_view and len(visible_satellites)>5000:
            # Sample complete orbital planes. Satellite-id stride sampling removed
            # nearly every along-track neighbour (id difference 1), while retaining
            # many cross-plane pairs (id difference per_plane), creating a false
            # impression that multi-shell constellations had no same-plane ISLs.
            plane_ids=sorted({row[1] for row in frame["satellites"] if row[0] in visible_satellites})
            satellites_by_plane={plane:[] for plane in plane_ids}
            for row in frame["satellites"]:
                if row[0] in visible_satellites:satellites_by_plane[row[1]].append(row[0])
            selected_planes=[];selected_count=0
            plane_stride=max(1,math.ceil(len(visible_satellites)/5000))
            for offset in range(0,len(plane_ids),plane_stride):
                # Two adjacent planes retain both same-plane rings and a useful
                # sample of within-shell cross-plane links.
                for plane in plane_ids[offset:offset+2]:
                    count_in_plane=len(satellites_by_plane[plane])
                    if selected_count+count_in_plane>6000:break
                    selected_planes.append(plane);selected_count+=count_in_plane
            selected_plane_set=set(selected_planes)
            visible_satellites={sid for plane in selected_plane_set for sid in satellites_by_plane[plane]}
        if page==1:
            related={int(item[0]) for row in frame.get("traffic",[]) for item in row[6]};visible_satellites&=related
        elif page==2:
            planning=frame.get("planning") or {};related=set()
            # Keep business-source satellites pickable on the planning map.
            # Their intermediate ISL path is fetched only after a click.
            related.update(int(row[0]) for row in frame.get("satellite_traffic",[]) if float(row[1])>0)
            for plan in [planning.get("active_plan")]+list(planning.get("plans") or []):
                if not plan:continue
                related.update(int(link.get("satellite")) for link in plan.get("links",[]) if link.get("satellite") is not None)
            active=planning.get("active_evaluation") or {}
            related.update(int(link.get("satellite")) for link in active.get("active_links",[]) if link.get("satellite") is not None)
            if related:visible_satellites&=related
            # Preserve the identity of every planned feeder on the map for
            # the full five-minute interval, even across display-plane filters.
            source_plan=planning.get("active_plan") or {}
            visible_satellites.update(int(link["satellite"]) for link in source_plan.get("links",[]) if link.get("satellite") is not None)
        filtered = {k:v for k,v in frame.items() if not k.startswith("_") and k not in ("satellites","isls","seams","gsls","traffic","satellite_traffic")}
        filtered["satellites"] = [row for row in frame["satellites"] if row[0] in visible_satellites]
        filtered["isls"] = [row for row in frame["isls"]
                            if row[0] in visible_satellites and row[1] in visible_satellites]
        if page==0 and len(filtered["isls"])>12000:
            step=math.ceil(len(filtered["isls"])/12000);filtered["isls"]=filtered["isls"][::step]
        filtered["seams"] = [row for row in frame["seams"]
                             if row[0] in visible_satellites and row[1] in visible_satellites]
        filtered["gsls"] = [row for row in frame["gsls"] if row[1] in visible_satellites]
        filtered["traffic"] = frame.get("traffic",[]) if page==1 else []
        if page==1 and map_bounds:
            west,south,east,north=map_bounds
            filtered["traffic"]=[row for row in filtered["traffic"] if south<=row[1]<=north and (west<=row[2]<=east if west<=east else row[2]>=west or row[2]<=east)]
        hotspot_cells={};hotspot_clusters={};hotspot_load={}
        for traffic_row in frame.get("traffic",[]):
            if not bool(traffic_row[4]):continue
            cluster_id=int(traffic_row[9]) if len(traffic_row)>9 else -1
            for allocation in traffic_row[6]:
                sid=int(allocation[0]);hotspot_cells[sid]=hotspot_cells.get(sid,0)+1;hotspot_load[sid]=hotspot_load.get(sid,0.)+float(allocation[1])
                if cluster_id>=0:hotspot_clusters.setdefault(sid,set()).add(cluster_id)
        filtered["satellite_traffic"] = ([list(row)+[hotspot_cells.get(int(row[0]),0),len(hotspot_clusters.get(int(row[0]),set())),round(hotspot_load.get(int(row[0]),0.),6)] for row in frame.get("satellite_traffic",[]) if row[0] in visible_satellites] if page in (1,2) else [])
        if page!=0:filtered["isls"]=[];filtered["seams"]=[];filtered["gsls"]=[]
        planning=frame.get("planning") if page==2 else None
        # The map needs the decision skeleton, not every business route.  Full
        # planning data is retained for result tabs and result-file export.
        if planning and options.get("subpage",0)==0:
            route_limit=0
            def compact_routes(rows):
                return []
            planning=dict(planning)
            source_plan=planning.get("active_plan") or (planning.get("plans") or [{}])[0]
            map_plan=dict(source_plan);map_plan["routes"]=compact_routes(source_plan.get("routes",[]))
            source_run=planning.get("active_evaluation") or {}
            map_run=dict(source_run);map_run["routes"]=compact_routes(source_run.get("routes",[]))
            planning["active_plan"]=map_plan;planning["plans"]=[map_plan]
            planning["active_evaluation"]=map_run;planning["evaluations"]=[map_run]
            planning["map_route_limit"]=route_limit
            planning["map_routes_truncated"]=len(source_run.get("routes",[]))>len(map_run["routes"])
        filtered["planning"]=planning
        filtered["display_page"]=page
        sat_by_id={row[0]:row for row in frame["satellites"]}; gateway_links=[[] for _ in range(options["station_count"])];links_by_sat={}
        for gid,sid,elevation in frame["gsls"]:
            if 0<=gid<len(gateway_links):gateway_links[gid].append([sid,elevation])
            links_by_sat.setdefault(sid,[]).append([gid,elevation])
        visible_rows=[]
        for sid,links in links_by_sat.items():
            sat=sat_by_id[sid]
            visible_rows.append([sid,sat[1],sat[2],sat[3],sat[4],links])
        visible_rows.sort(key=lambda row:max(x[1] for x in row[5]),reverse=True)
        elevations=[row[2] for row in frame["gsls"]]
        filtered["network_summary"]={"visible_satellite_count":len(visible_rows),"gsl_count":len(frame["gsls"]),"active_gateway_count":sum(bool(x) for x in gateway_links),"average_elevation_deg":sum(elevations)/len(elevations) if elevations else 0,"gateway_links":gateway_links,"visible_satellites":visible_rows[:20]}
        return filtered
    def _plane_filter_changed(self, _value=None):
        self._sync_worker_display_options()
        if self.worker is None and self.current_full_frame is not None:self._schedule_frame_filter(self.current_full_frame)

    @Slot(int,int)
    def _map_plane_filter_changed(self,center_plane,plane_count):
        self.center_plane.blockSignals(True); self.display_planes.blockSignals(True)
        if self.scenario_planes:
            self.center_plane.setValue(max(1,min(self.scenario_planes,center_plane+1)))
        self.display_planes.setCurrentIndex({3:0,5:1,0:2}.get(plane_count,0))
        self.center_plane.blockSignals(False); self.display_planes.blockSignals(False)
        self._plane_filter_changed()
    @Slot(float,float,float,float)
    def _map_viewport_changed(self,west,south,east,north):
        bounds=(round(west,3),round(south,3),round(east,3),round(north,3))
        if bounds==self.map_view_bounds:return
        self.map_view_bounds=bounds
        self._sync_worker_display_options()
        if self.worker is None and self.current_full_frame is not None:self._schedule_frame_filter(self.current_full_frame)
    @Slot(str)
    def _traffic_details_requested(self,cell_id):
        frame=self.current_full_frame
        if not frame:return
        row=next((item for item in frame.get("traffic",[]) if str(item[0])==str(cell_id)),None)
        if row is None:return
        payload=json.dumps(row[8] if len(row)>8 else [],separators=(',',':'))
        self.web.page().runJavaScript(f"setTrafficDetails({json.dumps(str(cell_id))},{payload})")
    @staticmethod
    def _compare_planned_exits(frame,planned_links,active_links,source,demand,current_route):
        """On-demand diagnostic; never runs in the per-second simulation loop."""
        positions=frame.get("satellites",[])
        source=int(source)
        if source<0 or source>=len(positions) or not planned_links:return [],"本帧没有规划出口"
        graph=_graph(positions,frame.get("isls",[]))
        targets={int(link.get("satellite",-1)) for link in planned_links}
        distance={source:0.};hops={source:0};queue=[(0.,source)];reached={}
        while queue and len(reached)<len(targets):
            length,node=heapq.heappop(queue)
            if length!=distance.get(node):continue
            if node in targets:reached[node]=(length,hops[node])
            for neighbor,edge_length in graph[node]:
                candidate=length+edge_length
                if candidate<distance.get(neighbor,math.inf):
                    distance[neighbor]=candidate;hops[neighbor]=hops[node]+1
                    heapq.heappush(queue,(candidate,neighbor))
        current_feeder=int(current_route.get("feeder",-1)) if current_route else -1
        candidates=[]
        active_by_key={(int(link.get("gateway",-1)),int(link.get("satellite",-1))):link for link in active_links}
        for planned in planned_links:
            gateway=int(planned.get("gateway",-1));feeder=int(planned.get("satellite",-1))
            link=active_by_key.get((gateway,feeder))
            if feeder not in reached:
                candidates.append({"gateway":gateway,"feeder":feeder,"hops":None,"isl_km":None,"gsl_km":None,
                    "delay_ms":None,"spare_gbps":0.,"current":feeder==current_feeder,"enough":False,"status":"ISL不可达"})
                continue
            isl_m,hop_count=reached[feeder]
            if link is None:
                candidates.append({"gateway":gateway,"feeder":feeder,"hops":hop_count,"isl_km":round(isl_m/1000,1),
                    "gsl_km":None,"delay_ms":None,"spare_gbps":0.,"current":feeder==current_feeder,
                    "enough":False,"status":"馈电链路不可用"})
                continue
            gsl_km=max(0.,float(link.get("gsl_distance_km",0)))
            rate=max(0.,float(link.get("rate_gbps",0)))
            used=max(0.,float(link.get("scheduled_gbps",0)))
            spare=max(0.,rate-used)
            candidates.append({"gateway":gateway,"feeder":feeder,
                "hops":hop_count,"isl_km":round(isl_m/1000,1),"gsl_km":round(gsl_km,1),
                "delay_ms":round((isl_m/1000+gsl_km)/299.792458,3),
                "spare_gbps":round(spare,4),"current":feeder==current_feeder,
                "enough":spare+1e-6>=demand,"status":"馈电余量足够" if spare+1e-6>=demand else "馈电余量不足"})
        candidates.sort(key=lambda item:(item["delay_ms"] is None,item["delay_ms"] if item["delay_ms"] is not None else math.inf,-item["spare_gbps"]))
        displayed=candidates[:5]
        gateways=sorted({item["gateway"] for item in candidates})
        for gateway in gateways:
            station_rows=[item for item in candidates if item["gateway"]==gateway]
            representative=next((item for item in station_rows if item["enough"]),station_rows[0])
            representative["station_best"]=True
            if representative not in displayed:displayed.append(representative)
        current=next((item for item in candidates if item["current"]),None)
        if current is not None and current not in displayed:displayed.append(current)
        viable=next((item for item in candidates if item["enough"] and not item["current"]),None)
        actual_delay=float(current_route.get("propagation_delay_ms",0)) if current_route else math.inf
        if current is not None and current["delay_ms"] is not None and current["delay_ms"]+0.5<actual_delay:
            conclusion="同一馈电出口存在更短的当前 ISL 路径；现行有效旧路径未主动重选"
        elif viable is None:conclusion="没有其他可达且余量足以承载本星全部业务的规划出口"
        elif current is None:conclusion="存在可用规划出口，但本星当前未走该出口；请核查未送达原因"
        elif viable["delay_ms"]+0.5<actual_delay:
            conclusion="存在更短且余量足够的出口；当前调度保留了仍有效的原路径"
        else:conclusion="当前路径与可承载的最近出口相比没有明显绕行"
        return displayed,conclusion+"；绿色站星线仅表示该馈电链路有余量，不表示本星到出口的 ISL 路径更短或沿途容量充足"
    @Slot(int,int)
    def _satellite_exit_comparison_requested(self,satellite_id,frame_index):
        self._satellite_path_requested(satellite_id,frame_index,compare_exits=True)
    @Slot(int,int)
    def _satellite_path_requested(self,satellite_id,frame_index,compare_exits=False):
        # The worker may be several frames ahead of the map. Route, geometry,
        # elevation and delay must all come from the frame the user clicked.
        frame=self.frame_cache.get(int(frame_index))
        if frame is None and self.current_full_frame is not None and int(self.current_full_frame.get("index",-1))==int(frame_index):
            frame=self.current_full_frame
        # Planning-map frames omit ISL/GSL/traffic rows to keep Cesium responsive.
        # Satellite details must use the complete frame for the same instant.
        if frame is not None and frame.get("display_page") is not None:
            full=None
            if self.current_full_frame is not None and int(self.current_full_frame.get("index",-1))==int(frame_index):
                full=self.current_full_frame
            if full is None and self.worker is not None:
                with self.worker._control_lock:
                    latest=self.worker._latest_full_frame
                    if latest is not None and int(latest.get("index",-1))==int(frame_index):full=latest
            if full is None and self.result_store is not None:
                full=self.result_store.read_frame(int(frame_index))
            if full is None:return
            frame=full
        if not frame:return
        planning=frame.get("planning") or {};execution=planning.get("execution_plan") or planning.get("active_plan") or {}
        run=planning.get("active_evaluation") or {}
        active_routes=[row for row in run.get("routes",[]) if int(row.get("source",-1))==int(satellite_id)]
        planned_route=next((row for row in execution.get("routes",[]) if int(row.get("source",-1))==int(satellite_id)),None)
        # A source may split traffic over several planned exits.  The map shows
        # the route carrying the most traffic, not an arbitrary list entry.
        route=max(active_routes,key=lambda row:float(row.get("demand_gbps",0))) if active_routes else planned_route
        demand_row=next((row for row in frame.get("satellite_traffic",[]) if int(row[0])==int(satellite_id)),None)
        demand=float(demand_row[1]) if demand_row else 0.;landing_demand=demand*float(self.active_config.traffic_landing_percent)/100. if self.active_config else demand
        served_cells=int(demand_row[2]) if demand_row and len(demand_row)>2 else 0
        service_cells=[];covered_cells=0;hotspot_clusters=set()
        for row in frame.get("traffic",[]):
            candidates=row[8] if len(row)>8 else []
            if any(int(candidate[0])==int(satellite_id) for candidate in candidates):covered_cells+=1
            allocation=next((item for item in (row[6] if len(row)>6 else []) if int(item[0])==int(satellite_id)),None)
            if allocation is None:continue
            cluster_id=int(row[9]) if len(row)>9 and row[9] is not None else -1
            if bool(row[4]) and cluster_id>=0:hotspot_clusters.add(cluster_id)
            service_cells.append({"id":str(row[0]),"lat":round(float(row[1]),4),"lon":round(float(row[2]),4),
                                  "allocated_gbps":round(float(allocation[1]),6),"demand_gbps":round(float(row[3]),6),
                                  "cluster_id":cluster_id if bool(row[4]) else -1})
        service_cells.sort(key=lambda item:item["allocated_gbps"],reverse=True)
        same_neighbors=[];cross_neighbors=[]
        for edge in frame.get("isls",[]):
            a,b,kind=int(edge[0]),int(edge[1]),int(edge[2])
            if a!=int(satellite_id) and b!=int(satellite_id):continue
            (same_neighbors if kind==0 else cross_neighbors).append(b if a==int(satellite_id) else a)
        station_links=[{"gateway":int(row[0]),"elevation_deg":round(float(row[2]),3)} for row in frame.get("gsls",[]) if int(row[1])==int(satellite_id)]
        station_links.sort(key=lambda item:item["elevation_deg"],reverse=True)
        path=[int(x) for x in route.get("path",[])] if route else []
        edge_types={(min(int(a),int(b)),max(int(a),int(b))):int(kind) for a,b,kind,*_ in frame.get("isls",[])}
        active_feeders={int(x.get("satellite",-1)) for x in run.get("active_links",[])}
        planned_feeders={int(x.get("satellite",-1)) for x in execution.get("links",[])}
        broken=[(a,b) for a,b in zip(path,path[1:]) if (min(a,b),max(a,b)) not in edge_types]
        feeder=int(route.get("feeder",path[-1] if path else -1)) if route else -1
        feeder_active=feeder in active_feeders;available=bool(active_routes) and feeder_active and not broken
        if demand<=0:reason="no_business"
        elif not route:reason="no_reachable_feeder"
        elif not feeder_active:reason="feeder_link_unavailable"
        elif broken:reason="isl_interrupted"
        elif not active_routes:reason="feeder_capacity_exhausted"
        else:reason="available"
        detail={"frame":frame.get("index",-1),"source":int(satellite_id),"available":available,"reason":reason,
                 "has_route":bool(route),"demand_gbps":round(landing_demand,6),"access_business_gbps":round(demand,6),
                 "delivered_gbps":round(sum(float(row.get("demand_gbps",0)) for row in active_routes),6),
                 "shown_route_gbps":round(float(route.get("demand_gbps",0)),6) if active_routes else 0.,
                 "active_route_count":len(active_routes),"served_cells":served_cells,
                "covered_cells":covered_cells,"hotspot_clusters":len(hotspot_clusters),"service_cells":service_cells,
                "is_active_feeder":int(satellite_id) in active_feeders,"is_planned_feeder":int(satellite_id) in planned_feeders,
                "same_neighbors":sorted(same_neighbors),"cross_neighbors":sorted(cross_neighbors),"station_links":station_links}
        if compare_exits:
            comparison,conclusion=self._compare_planned_exits(frame,execution.get("links",[]),run.get("active_links",[]),satellite_id,landing_demand,route if active_routes else None)
            detail["exit_comparison"]=comparison;detail["exit_comparison_conclusion"]=conclusion
        if route:
            position_by_id={int(row[0]):row for row in frame.get("satellites",[])}
            points=[[sid,position_by_id[sid][2],position_by_id[sid][3],position_by_id[sid][4]] for sid in path if sid in position_by_id]
            feeder_link=next((x for x in run.get("active_links",[]) if int(x.get("satellite",-1))==feeder),None)
            segments=[]
            for index,(a,b) in enumerate(zip(path,path[1:])):
                row_a=position_by_id.get(a);row_b=position_by_id.get(b);key=(min(a,b),max(a,b))
                distance_km=math.dist(geodetic_to_ecef(row_a[2],row_a[3],row_a[4]),geodetic_to_ecef(row_b[2],row_b[3],row_b[4]))/1000 if row_a and row_b else 0.
                segments.append({"index":index+1,"from":a,"to":b,"type":("same_plane" if edge_types.get(key)==0 else ("cross_plane" if edge_types.get(key)==1 else "broken")),
                                 "available":key in edge_types,"distance_km":round(distance_km,3),"delay_ms":round(distance_km/299.792458,4)})
            gsl=None
            if feeder_link:
                station=frame.get("stations",[])[int(feeder_link["gateway"])] if int(feeder_link["gateway"])<len(frame.get("stations",[])) else self.active_config.stations[int(feeder_link["gateway"])]
                feeder_row=position_by_id.get(feeder);gsl_distance=math.dist(geodetic_to_ecef(feeder_row[2],feeder_row[3],feeder_row[4]),geodetic_to_ecef(station["lat"],station["lon"],station.get("alt",0)))/1000 if feeder_row else 0.
                gsl={"gateway":int(feeder_link["gateway"]),"elevation_deg":float(feeder_link.get("elevation_deg",0)),"distance_km":round(gsl_distance,3),
                     "delay_ms":round(gsl_distance/299.792458,4),
                     "rain_db":float(feeder_link.get("rain_actual_db",0)),"snr_db":float(feeder_link.get("snr_db",0)),"modcod":str(feeder_link.get("modcod","--")),
                     "rate_gbps":float(feeder_link.get("rate_gbps",0)),"scheduled_gbps":float(feeder_link.get("scheduled_gbps",0))}
            detail.update(path=path,points=points,feeder=feeder,gateway=(int(feeder_link["gateway"]) if feeder_link else None),
                          hops=int(route.get("hops",max(0,len(path)-1))),distance_km=float(route.get("distance_km",0)),
                          propagation_delay_ms=float(route.get("propagation_delay_ms",sum(x["delay_ms"] for x in segments)+(gsl["delay_ms"] if gsl else 0))),segments=segments,gsl=gsl)
        payload=json.dumps(detail,ensure_ascii=False,separators=(',',':'))
        self.web.page().runJavaScript(f"setSatellitePath({int(satellite_id)},{payload})")
    def _layer_changed(self, _checked=None):
        if not self.web_ready:
            return
        self.web.page().runJavaScript(f"setDisplayOptions({json.dumps(self._layer_options())})")
    def _layer_options(self):
        return {"satellites":self.show_sat.isChecked(),"same":self.show_same.isChecked(),
                "cross":self.show_cross.isChecked(),"gsl":self.show_gsl.isChecked(),
                "motion":self.show_motion.isChecked(),"labels":self.show_labels.isChecked(),
                "traffic":True,"serviceAreas":True,"trafficRegions":True,"trafficCoverage":False,
                "trafficLinks":True,"seam":True,"latitude":True}
    def _start_transfer(self):
        if self.pending_frame is None:return
        frame=self.pending_frame; self.pending_frame=None
        self.transfer_frame_index=frame.get("index",-1);self.transfer_started_at=time.perf_counter()
        self.transfer_frame_time=frame.get("time_s",0);self.transfer_absolute_time=frame.get("absolute_time","--")
        page=int(frame.get("display_page",0)); current_isls={self._isl_transfer_key(row):row for row in frame.get("isls",[])}
        use_isl_delta=page==0 and self.last_transfer_page==0
        if use_isl_delta:
            isl_add=[row for key,row in current_isls.items() if key not in self.last_sent_isls]
            isl_remove=[key for key in self.last_sent_isls if key not in current_isls]
        else:
            isl_add=[]; isl_remove=[]
        self.transfer_row_counts={name:len(frame.get(name,[])) for name in ("satellites","isls","seams","gsls","traffic","satellite_traffic")}
        self.transfer_row_counts.update({"isl_add":len(isl_add),"isl_remove":len(isl_remove),"isl_retained":len(current_isls)-len(isl_add) if use_isl_delta else 0})
        head={k:v for k,v in frame.items() if k not in ("satellites","isls","seams","gsls","traffic","satellite_traffic")}
        head["isl_delta"]=use_isl_delta
        commands=[f"beginFrame({json.dumps(head,separators=(',',':'))})"]
        transfer_rows={name:frame[name] for name in ("satellites","seams","gsls")}
        coverage_counts={}
        for row in frame["traffic"]:
            for candidate in row[8]:coverage_counts[int(candidate[0])]=coverage_counts.get(int(candidate[0]),0)+1
        transfer_rows["satellite_traffic"]=[[row[0],row[1],row[2],coverage_counts.get(int(row[0]),0)]+list(row[3:]) for row in frame["satellite_traffic"]]
        if self.prepared_metadata and self.prepared_metadata.get("traffic_cells"):
            compact=[]
            for row in frame["traffic"]:
                elevations={int(item[0]):item[1] for item in row[8]}
                allocations=[[item[0],item[1],elevations.get(int(item[0]),0.0)] for item in row[6]]
                compact.append([row[0],row[3],row[4],row[5],allocations,row[7],row[9],len(row[8])])
            transfer_rows["traffic_dynamic"]=compact
        else:
            transfer_rows["traffic"]=frame["traffic"]
        if use_isl_delta:
            transfer_rows["isls_add"]=isl_add;transfer_rows["isls_remove"]=isl_remove
        else:transfer_rows["isls"]=frame["isls"]
        for name,size in (("satellites",5000),("isls",10000),("isls_add",10000),("isls_remove",10000),("seams",5000),("gsls",5000),("traffic",2000),("traffic_dynamic",2000),("satellite_traffic",2000)):
            rows=transfer_rows.get(name,[])
            for i in range(0,len(rows),size):commands.append(f"appendFrameData('{name}',{json.dumps(rows[i:i+size],separators=(',',':'))})")
        self.last_transfer_page=page
        self.last_sent_isls=current_isls if page==0 else {}
        commands.append("finishFrame()")
        self.transfer_queue=commands; self.transfer_progress.setMaximum(len(commands)); self.transfer_progress.setValue(0); self._send_next()
    @staticmethod
    def _isl_transfer_key(row):
        a,b=int(row[0]),int(row[1]);return f"{min(a,b)}:{max(a,b)}:{int(row[2]) if len(row)>2 else 0}"
    def _send_next(self):
        if self.transfer_in_flight:return
        if not self.transfer_queue:
            if self.pending_frame is not None:self._start_transfer()
            return
        command=self.transfer_queue.pop(0)
        done=self.transfer_progress.maximum()-len(self.transfer_queue);self.transfer_in_flight=True;self.web.page().runJavaScript(command,lambda _result:self._transfer_done(done))
    def _transfer_done(self,value):
        self.transfer_in_flight=False;self.transfer_progress.setValue(value)
        if not self.transfer_queue and self.active_config:
            if not self.timeline_dragging:
                self.timeline.setValue(getattr(self,"transfer_frame_index",0));self.timeline_label.setText(f't={getattr(self,"transfer_frame_time",0):g} s');self.absolute_time_label.setText(f"UTC：{getattr(self,'transfer_absolute_time','--').replace('+00:00','Z')}")
            append_diagnostic(str(Path(self.active_config.result_directory)/"time_variation.jsonl") if self.active_config.result_directory else "","frame_render_transfer_finished",frame=getattr(self,"transfer_frame_index",-1),transfer_ms=round((time.perf_counter()-getattr(self,"transfer_started_at",time.perf_counter()))*1000,1),rows=getattr(self,"transfer_row_counts",{}),replacement_waiting=self.pending_frame is not None)
        QTimer.singleShot(0,self._send_next)

    def _save_simulation_data(self):
        tables=(
            ("调度执行_执行时序",self.planning_resources.history),
            ("调度执行_信关站资源",self.planning_resources.deployment),
            ("调度执行_馈电链路",self.planning_resources.links),
            ("效能评估_关键事件",self.planning_evaluation.events),
        )
        if not any(table.rowCount() for _,table in tables) and not self.planning_evaluation.chart.rows:
            QMessageBox.information(self,"保存仿真数据","当前还没有调度执行或效能评估数据。")
            return
        root=QFileDialog.getExistingDirectory(self,"选择仿真数据保存位置",str(ROOT))
        if not root:return
        output=Path(root)/f"仿真数据_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            output.mkdir(parents=True,exist_ok=False)
            (output/"调度执行_摘要.txt").write_text(self.planning_resources.summary.text(),encoding="utf-8")
            (output/"效能评估_摘要.txt").write_text(self.planning_evaluation.summary.text(),encoding="utf-8")
            for name,table in tables:
                columns=[column for column in range(table.columnCount()) if not table.isColumnHidden(column)]
                with (output/f"{name}.csv").open("w",encoding="utf-8-sig",newline="") as stream:
                    writer=csv.writer(stream)
                    writer.writerow([table.horizontalHeaderItem(column).text() if table.horizontalHeaderItem(column) else f"第{column+1}列" for column in columns])
                    for row in range(table.rowCount()):
                        writer.writerow([table.item(row,column).text() if table.item(row,column) else "" for column in columns])
            for name,headers,chart in (
                ("效能评估_业务容量",["时刻 s","落地需求 Gb/s","已送达 Gb/s","可用链路容量 Gb/s"],self.planning_evaluation.chart),
                ("效能评估_传播时延",["时刻 s","加权平均时延 ms","P95时延 ms"],self.planning_evaluation.delay_chart),
            ):
                with (output/f"{name}.csv").open("w",encoding="utf-8-sig",newline="") as stream:
                    writer=csv.writer(stream);writer.writerow(headers);writer.writerows(chart.rows)
            if not self.planning_evaluation.save_charts_image(output/"效能评估_趋势图.png"):
                raise OSError("效能评估趋势图保存失败")
            self.status.setText(f"仿真数据已保存到 {output}")
            QMessageBox.information(self,"保存仿真数据",f"已保存到：\n{output}")
        except Exception as exc:
            QMessageBox.critical(self,"保存仿真数据失败",f"{type(exc).__name__}: {exc}")

    def _export_all_result_tables(self):
        tables=(
            ("规划方案_规划批次",self.planning_results.plans),
            ("规划方案_馈电链路",self.planning_results.plan_links),
            ("调度执行_执行时序",self.planning_resources.history),
            ("调度执行_信关站资源",self.planning_resources.deployment),
            ("调度执行_馈电链路",self.planning_resources.links),
            ("效能评估_关键事件",self.planning_evaluation.events),
        )
        if not any(table.rowCount() for _name,table in tables):
            QMessageBox.information(self,"导出全部表格","当前还没有可导出的表格数据。")
            return
        root=QFileDialog.getExistingDirectory(self,"选择全部表格的导出位置",str(ROOT))
        if not root:return
        output=Path(root)/f"规划调度表格_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            output.mkdir(parents=True,exist_ok=False)
            index_rows=[]
            for name,table in tables:
                path=output/f"{name}.csv"
                if name=="规划方案_馈电链路" and self.planning_results.offline_catalog:
                    headers=["生效时刻","雨衰工况","信关站","规划馈电卫星","5分钟采样最低仰角","规划采用雨衰 dB","Es/N0 dB","调制编码","规划容量 Gb/s","雨衰等级","TLE名称","TLE第1行","TLE第2行"]
                    rows=[]
                    for plan in self.planning_results.offline_catalog:
                        valid_from=float(plan.get("valid_from_s",0));time_text=f"{int(valid_from//60):02d}:{int(valid_from%60):02d}"
                        for link in plan.get("links",[]):
                            rows.append([time_text,plan.get("rain_grade","--"),f"GW-{int(link.get('gateway',0))+1:02d}",
                                f"SAT-{int(link.get('satellite',0))+1:05d}",f"{float(link.get('elevation_deg',0)):.2f}°",
                                f"{float(link.get('rain_attenuation_db',0)):.2f}",f"{float(link.get('snr_db',0)):.2f}",link.get("modcod","--"),
                                f"{float(link.get('rate_gbps',0)):.3f}",_rain_severity_text(link.get("rain_severity","--")),
                                (link.get("tle") or {}).get("name",""),(link.get("tle") or {}).get("line1",""),(link.get("tle") or {}).get("line2","")])
                    with path.open("w",encoding="utf-8-sig",newline="") as stream:
                        writer=csv.writer(stream);writer.writerow(headers);writer.writerows(rows)
                    index_rows.append([name,len(rows),path.name]);continue
                columns=[column for column in range(table.columnCount()) if not table.isColumnHidden(column)]
                with path.open("w",encoding="utf-8-sig",newline="") as stream:
                    writer=csv.writer(stream)
                    writer.writerow([(table.horizontalHeaderItem(column).text() if table.horizontalHeaderItem(column) else f"第{column+1}列") for column in columns])
                    for row in range(table.rowCount()):writer.writerow([(table.item(row,column).text() if table.item(row,column) else "") for column in columns])
                index_rows.append([name,table.rowCount(),path.name])
            with (output/"表格清单.csv").open("w",encoding="utf-8-sig",newline="") as stream:
                writer=csv.writer(stream);writer.writerow(["表格名称","数据行数","文件名称"]);writer.writerows(index_rows)
            QMessageBox.information(self,"导出完成",f"全部表格已保存到：\n{output}")
            self.status.setText(f"全部表格已导出到 {output}")
        except Exception as exc:
            QMessageBox.critical(self,"导出失败",f"{type(exc).__name__}: {exc}")

    def export_basic_data(self):
        """导出生成基础数据阶段产生的离线资产，不依赖动态仿真帧。"""
        if self.active_config is None or self.prepared_metadata is None:
            QMessageBox.information(self,"尚无基础数据","请先点击“生成基础数据”，等待离线场景建立完成。")
            return
        folder=QFileDialog.getExistingDirectory(self,"选择基础数据导出目录",str(ROOT))
        if not folder:return
        out=Path(folder);cfg=self.active_config;prepared=self.prepared_metadata
        try:
            (out/"scenario_config.json").write_text(json.dumps(asdict(cfg),ensure_ascii=False,indent=2),encoding="utf-8")
            manifest={"asset_type":"offline_basic_data","generated_at":datetime.now().astimezone().isoformat(),
                      "satellite_count":prepared.get("satellite_count",0),"planes":prepared.get("planes",0),
                      "per_plane":prepared.get("per_plane",0),"isl_candidate_count":prepared.get("isl_candidate_count",0),
                      "seam_candidate_count":prepared.get("seam_candidate_count",0),"gateway_count":len(cfg.stations),
                      "traffic_cell_count":prepared.get("traffic_cell_count",0),"epoch":prepared.get("epoch"),
                      "contains_dynamic_frames":False,"contains_orbit_position_cache":bool(prepared.get("orbit_cache"))}
            orbit_cache_source=Path(str(prepared.get("orbit_cache_path","")))
            if prepared.get("orbit_cache") and orbit_cache_source.is_file():
                shutil.copy2(orbit_cache_source,out/"orbit_positions.npy");manifest["orbit_position_cache"]="orbit_positions.npy";manifest["orbit_cache_size_mb"]=prepared.get("orbit_cache_size_mb",0)
            tle_asset_source=Path(str(prepared.get("tle_asset_path","")))
            if tle_asset_source.is_file():
                shutil.copy2(tle_asset_source,out/"satellites.tles");manifest["satellite_tles"]="satellites.tles";manifest["tle_asset_hash"]=prepared.get("tle_asset_hash","")
            traffic_cache_source=Path(str(prepared.get("traffic_cache_path","")))
            if prepared.get("traffic_cache") and traffic_cache_source.is_file():
                source_db=sqlite3.connect(str(traffic_cache_source));target_db=sqlite3.connect(str(out/"traffic_timeseries.sqlite"));source_db.backup(target_db);target_db.close();source_db.close();manifest["traffic_timeseries_cache"]="traffic_timeseries.sqlite"
            topology_cache_source=Path(str(prepared.get("topology_cache_path","")))
            if topology_cache_source.is_file():
                shutil.copy2(topology_cache_source,out/"isl_candidates.npz");manifest["isl_candidates_cache"]="isl_candidates.npz"
            isl_model_source=Path(str(prepared.get("isl_model_path","")))
            if prepared.get("isl_model_path") and isl_model_source.is_file():
                shutil.copy2(isl_model_source,out/"isl_model.json");manifest["isl_model"]="isl_model.json"
            rain_cache_source=Path(str(prepared.get("rain_cache_path","")))
            if prepared.get("rain_cache_path") and rain_cache_source.is_file():
                shutil.copy2(rain_cache_source,out/"rain_spatial_factors.json");manifest["rain_spatial_cache"]="rain_spatial_factors.json"
            rain_model_source=Path(str(prepared.get("rain_model_path","")))
            if prepared.get("rain_model_path") and rain_model_source.is_file():
                shutil.copy2(rain_model_source,out/"rain_model.json");manifest["rain_model"]="rain_model.json"
            offline_plan_source=Path(cfg.scenario_directory)/"base_data"/"offline_plans.json" if cfg.scenario_directory else Path()
            if cfg.scenario_directory and offline_plan_source.is_file():
                shutil.copy2(offline_plan_source,out/"offline_plans.json")
                try:
                    offline_data=json.loads(offline_plan_source.read_text(encoding="utf-8"));stats=offline_data.get("statistics") or {}
                    manifest["offline_planning"]={"file":"offline_plans.json","planning_spec_hash":offline_data.get("planning_spec_hash",""),"plan_count":len(offline_data.get("plans",[])),**stats}
                except (OSError,ValueError,TypeError):manifest["offline_planning"]={"file":"offline_plans.json"}
            (out/"basic_assets_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
            with (out/"gateways.csv").open("w",newline="",encoding="utf-8-sig") as f:
                w=csv.writer(f);w.writerow(["gateway_id","name","latitude_deg","longitude_deg","altitude_m","antenna_count","channel_count"])
                for gid,gs in enumerate(cfg.stations):w.writerow([gid,gs["name"],gs["lat"],gs["lon"],gs.get("alt",0),gs.get("antenna_count",20),gs.get("channel_count",gs.get("antenna_count",20))])
            (out/"gateway_deployment.json").write_text(json.dumps(prepared.get("gateway_deployment",{}),ensure_ascii=False,indent=2),encoding="utf-8")
            manifest["gateway_deployment"]="gateway_deployment.json"
            candidates=[];seams=[]
            if topology_cache_source.is_file():
                import numpy as np
                with np.load(topology_cache_source) as stored_topology:
                    candidates=stored_topology["candidates"].tolist();seams=stored_topology["seams"].tolist()
            else:
                shell_runs=prepared.get("walker_shells") or [{"planes":cfg.planes,"per_plane":max(1,int(prepared.get("per_plane",cfg.satellite_count//max(1,cfg.planes)))),"satellite_offset":0}]
                for shell in shell_runs:
                    local,local_seams=make_candidates(int(shell["planes"]),int(shell["per_plane"]),cfg.isl_pattern,cfg.seam_enabled,int(shell["planes"])-1)
                    offset=int(shell.get("satellite_offset",0));candidates.extend((a+offset,b+offset,k) for a,b,k in local);seams.extend((a+offset,b+offset,k) for a,b,k in local_seams)
            seam_pairs={(a,b) for a,b,_kind in seams}
            with (out/"isl_candidates.csv").open("w",newline="",encoding="utf-8-sig") as f:
                w=csv.writer(f);w.writerow(["source_satellite","target_satellite","link_type","seam_candidate"])
                for a,b,kind in candidates:w.writerow([a,b,"same_plane" if kind==0 else "cross_plane",False])
                for a,b,kind in seams:w.writerow([a,b,"cross_plane",(a,b) in seam_pairs])
            table=load_modcod_table(cfg.modcod_table_path) if cfg.modcod_table_path else DEFAULT_MODCOD_TABLE
            with (out/"modcod_table.csv").open("w",newline="",encoding="utf-8-sig") as f:
                w=csv.writer(f);w.writerow(["modcod","minimum_snr_db","spectral_efficiency","rate_gbps"])
                for row in table:w.writerow([row.name,row.minimum_snr_db,row.spectral_efficiency,"" if row.rate_gbps is None else row.rate_gbps])
            source_files=[]
            for raw,name in ((cfg.tle_path,"source_tles.txt"),(cfg.traffic_grid_path,"traffic_grid.csv"),(cfg.modcod_table_path,"source_modcod_table.csv")):
                if raw and Path(raw).is_file():shutil.copy2(raw,out/name);source_files.append(name)
            manifest["copied_source_files"]=source_files
            (out/"basic_assets_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
            QMessageBox.information(self,"基础数据导出完成",f"已导出到：\n{out}\n\n不包含时变仿真帧。")
            self.status.setText(f"基础数据已导出到 {out}")
        except Exception as exc:
            QMessageBox.critical(self,"导出基础数据失败",f"{type(exc).__name__}: {exc}")

    def export_results(self):
        if self.active_config is None or self.current_full_frame is None:
            QMessageBox.information(self,"暂无结果","请先生成场景并至少计算一个时间点。")
            return
        folder=QFileDialog.getExistingDirectory(self,"选择导出目录",str(ROOT))
        if not folder:return
        out=Path(folder); frame=self.current_full_frame; cfg=self.active_config
        try:
            (out/"scenario_config.json").write_text(json.dumps(asdict(cfg),ensure_ascii=False,indent=2),encoding="utf-8")
            station_names=[x["name"] for x in cfg.stations]
            summary_fields=["index","time_s","absolute_time","compute_ms","satellite_count","same_isl_count","cross_isl_count","seam_disabled_count","gsl_count","visible_satellite_count","active_gateway_count","average_elevation_deg","max_elevation_deg"]+[f"gateway_{i}_{name}_visible_count" for i,name in enumerate(station_names)]
            with (out/"time_series_summary.csv").open("w",newline="",encoding="utf-8-sig") as f:
                w=csv.DictWriter(f,fieldnames=summary_fields); w.writeheader()
                for item in (self.summary_history[k] for k in sorted(self.summary_history)):
                    row={key:item.get(key,"") for key in summary_fields}
                    for i,name in enumerate(station_names): row[f"gateway_{i}_{name}_visible_count"]=item["station_visible_counts"][i]
                    w.writerow(row)
            same_neighbors={}; cross_neighbors={}; sat_gateways={}
            for a,b,kind in frame["isls"]:
                target=same_neighbors if kind==0 else cross_neighbors
                target.setdefault(a,[]).append(b); target.setdefault(b,[]).append(a)
            for gid,sid,elevation in frame["gsls"]: sat_gateways.setdefault(sid,[]).append(f"{gid}:{station_names[gid]}:{elevation:.4f}")
            with (out/"current_satellites.csv").open("w",newline="",encoding="utf-8-sig") as f:
                w=csv.writer(f); w.writerow(["satellite_id","plane","slot","latitude_deg","longitude_deg","altitude_m","same_plane_neighbors","cross_plane_neighbors","visible_gateways(gid:name:elevation_deg)"])
                for sid,plane,lat,lon,alt in frame["satellites"]: w.writerow([sid,plane+1,sid%(len(frame["satellites"])//cfg.planes)+1,lat,lon,alt,";".join(map(str,same_neighbors.get(sid,[]))),";".join(map(str,cross_neighbors.get(sid,[]))),";".join(sat_gateways.get(sid,[]))])
            with (out/"current_isls.csv").open("w",newline="",encoding="utf-8-sig") as f:
                w=csv.writer(f); w.writerow(["source_satellite_id","target_satellite_id","type"]); w.writerows((a,b,"same_plane" if kind==0 else "cross_plane") for a,b,kind in frame["isls"])
            with (out/"current_gsls.csv").open("w",newline="",encoding="utf-8-sig") as f:
                w=csv.writer(f); w.writerow(["gateway_id","gateway_name","satellite_id","elevation_deg"]); w.writerows((gid,station_names[gid],sid,elevation) for gid,sid,elevation in frame["gsls"])
            with (out/"current_gateways.csv").open("w",newline="",encoding="utf-8-sig") as f:
                w=csv.writer(f); w.writerow(["gateway_id","name","latitude_deg","longitude_deg","altitude_m","antenna_count","channel_count","min_elevation_deg","max_gsl_km","visible_satellite_count","visible_satellite_ids"])
                for gid,gs in enumerate(cfg.stations):
                    ids=[sid for g,sid,_e in frame["gsls"] if g==gid]; w.writerow([gid,gs["name"],gs["lat"],gs["lon"],gs["alt"],gs.get("antenna_count",20),gs.get("channel_count",gs.get("antenna_count",20)),cfg.min_elevation_deg,cfg.max_gsl_km,len(ids),";".join(map(str,ids))])
            with (out/"current_traffic_cells.csv").open("w",newline="",encoding="utf-8-sig") as f:
                w=csv.writer(f); w.writerow(["cell_id","latitude_deg","longitude_deg","demand_gbps","dense_cluster_member","unassigned_gbps","allocations(satellite_id:gbps)","local_hour","visible_satellite_count","dense_cluster_id"])
                for row in frame.get("traffic",[]):
                    w.writerow([row[0],row[1],row[2],row[3],row[4],row[5],
                                ";".join(f"{sid}:{value:.6f}" for sid,value in row[6]),
                                row[7],len(row[8]),row[9] if len(row)>9 else -1])
            with (out/"current_satellite_traffic.csv").open("w",newline="",encoding="utf-8-sig") as f:
                w=csv.writer(f); w.writerow(["satellite_id","assigned_gbps","served_cell_count"])
                w.writerows(frame.get("satellite_traffic",[]))
            planning=frame.get("planning")
            if planning:
                with (out/"current_gateway_deployment.csv").open("w",newline="",encoding="utf-8-sig") as f:
                    w=csv.writer(f);w.writerow(["gateway_id","gateway_name","visible_satellites","planned_channels","used_channels","recommended_antennas","minimum_connection_satisfied","scope"])
                    for row in planning.get("gateway_deployment",{}).get("stations",[]):
                        w.writerow([row["gateway"],row["name"],row["visible_satellites"],row["planned_channels"],row["used_channels"],row["recommended_antennas"],row["minimum_connection_satisfied"],"current_site_set_evaluation"])
                with (out/"current_slow_loop_plan.csv").open("w",newline="",encoding="utf-8-sig") as f:
                    w=csv.writer(f); w.writerow(["plan_id","plan_name","is_active","planned_gsl_count","gateway_count","planned_capacity_gbps","demand_gbps","reachable_demand_gbps","average_isl_hops","business_weighted_hops","weighted_propagation_delay_ms","p95_propagation_delay_ms","maximum_propagation_delay_ms","latency_scope"])
                    for plan in planning.get("plans",[]):
                        w.writerow([plan["id"],plan["name"],plan["id"]==planning.get("active_plan_id"),len(plan["links"]),plan["gateway_count"],plan["total_capacity_gbps"],plan["total_demand_gbps"],plan["reachable_demand_gbps"],plan["average_hops"],plan.get("business_weighted_hops",0),plan.get("business_propagation_delay_ms",0),plan.get("p95_propagation_delay_ms",0),plan.get("maximum_propagation_delay_ms",0),plan.get("latency_scope","")])
                with (out/"current_planning_routes.csv").open("w",newline="",encoding="utf-8-sig") as f:
                    w=csv.writer(f);w.writerow(["plan_id","source_satellite","feeder_satellite","isl_hops","distance_km","demand_gbps","path"])
                    for plan in planning.get("plans",[]):
                        for route in plan.get("routes",[]):w.writerow([plan["id"],route["source"],route["feeder"],route["hops"],route["distance_km"],route["demand_gbps"],"->".join(map(str,route["path"]))])
                with (out/"current_schedule_links.csv").open("w",newline="",encoding="utf-8-sig") as f:
                    w=csv.writer(f); w.writerow(["gateway_id","gateway_name","satellite_id","elevation_deg","snr_db","modcod","array_mode_m","power_factor_rho","rate_gbps","scheduled_gbps","utilization","belongs_to_slow_plan","fast_loop_substitute"])
                    plan=planning.get("active_plan") or {};planned={(x["gateway"],x["satellite"]) for x in plan.get("links",[])}
                    for actual in (planning.get("active_evaluation") or {}).get("active_links",[]):
                        gid=actual["gateway"];pair=(gid,actual["satellite"])
                        w.writerow([gid,station_names[gid],actual["satellite"],actual["elevation_deg"],actual["snr_db"],actual["modcod"],actual.get("array_mode",1),actual.get("power_factor",1),actual["rate_gbps"],actual.get("scheduled_gbps",0),actual.get("utilization",0),pair in planned,actual.get("is_substitute",False)])
            QMessageBox.information(self,"导出完成",f"已导出到：\n{out}")
            self.status.setText(f"结果已导出到 {out}")
        except Exception as exc:
            QMessageBox.critical(self,"导出失败",str(exc))
    @Slot(int,str)
    def _progress(self,value,text):
        if self.user_stopping:return
        self.compute_progress.setValue(value);self.status.setText(text)
        if self.elapsed_phase=="preparation" or (self.offline_planning_started_at is not None and str(text).startswith("离线规划")):
            self.busy_overlay.set_progress(value,text)
        if "业务缓存" in text:self.cache_status_label.setText("业务缓存生成中")
        elif "轨道缓存" in text:self.cache_status_label.setText("轨道缓存生成中")
    @Slot(str)
    def _failed(self,text):
        if self.user_stopping:return
        self.busy_overlay.stop();self.offline_planning_started_at=None
        self._reset_one_click_run()
        if self.run_started_at is not None:self.run_elapsed_s+=max(0.0,time.perf_counter()-self.run_started_at);self.run_started_at=None
        if self._closing:return
        QMessageBox.critical(self,"仿真失败",text); self.status.setText(text); self.state_badge.setText("● 错误"); self.alert_label.setText("告警 1")
    @Slot()
    def _finished(self):
        if self.user_stopping:return
        self.busy_overlay.stop()
        self._reset_one_click_run()
        if self.run_started_at is not None:self.run_elapsed_s+=max(0.0,time.perf_counter()-self.run_started_at);self.run_started_at=None
        self.start_btn.setText("开始仿真"); self.start_btn.setEnabled(False); self.stop_btn.setEnabled(False); self.status.setText(self.status.text()+"；仿真结束"); self.state_badge.setText("● 已结束")
    def _thread_stopped(self):
        if self._closing:
            self.worker=None;self.thread=None
            QTimer.singleShot(0,self.close)
            return
        self.busy_overlay.stop();self.offline_planning_started_at=None
        restart_preparation=self.regenerate_pending;self.regenerate_pending=False
        restart_all=self._run_all_after_stop;self._run_all_after_stop=False
        self.worker = None; self.thread = None; self.prepared_ready = False; self.start_btn.setEnabled(False);self.prepare_btn.setText("生成基础数据");self.prepare_btn.setEnabled(True);self._populate_planning_choices([]); self._set_scenario_editable(True);self._refresh_parameter_actions()
        if self.user_stopping:
            saved_path=self.result_store_path
            if self.result_store:self.result_store.close(); self.result_store=None
            self.frame_cache.clear(); self.latest_full_frame = None; self.current_full_frame = None; self.pending_frame = None
            self.summary_history.clear(); self.active_config = None; self.prepared_metadata = None
            self.planning_results.clear();self.planning_resources.clear();self.planning_evaluation.clear()
            self.result_store_path=None;self.scenario_planes=None;self.offline_planning_started_at=None
            self.run_started_at=None;self.run_elapsed_s=0.0;self.preparation_started_at=None;self.elapsed_phase="idle"; self.elapsed_label.setText("准备耗时 00:00:00");self.cache_status_label.setText("缓存 --");self.cache_status_label.setToolTip("悬停查看各模型缓存状态"); self.state_badge.setText("● 就绪"); self.alert_label.setText("告警 0")
            self.transfer_queue.clear(); self.follow_live = True; self.is_paused = False
            self.start_btn.setText("开始仿真"); self.start_btn.setEnabled(False);self.stop_btn.setEnabled(False)
            self.timeline.setValue(0); self.timeline.setRange(0,0); self.timeline.setEnabled(False)
            self.timeline_label.setText("t=0 s"); self.absolute_time_label.setText("UTC：--")
            self.compute_progress.setValue(0); self.transfer_progress.setValue(0); self.status.setText(f"结果包已保存：{saved_path}" if saved_path else "等待开始")
            self._set_tree_children(self.scene_tree,[]);self._set_tree_children(self.isl_tree,[]);self._set_tree_children(self.gateway_tree,[]);self._set_tree_children(self.traffic_tree,[]);self._set_tree_children(self.rain_tree,[]);self._set_tree_children(self.results_tree,[])
            self.current_scene_node.setText(0,"当前场景（未生成）"); self._refresh_scene_preview()
            if self.web_ready: self.web.page().runJavaScript("resetScenario()")
            self.user_stopping = False
        self._reset_one_click_run()
        if restart_preparation:QTimer.singleShot(0,self.start_simulation)
        elif restart_all:QTimer.singleShot(0,self._run_all_stages)
    def stop_simulation(self):
        self._one_click_running=False;self._run_all_after_stop=False;self.regenerate_pending=False
        if self.loaded_result:
            self.result_play_timer.stop(); self.result_playing=False; self.loaded_result=False; self.loaded_frame_indices=[]
            if self.result_store:self.result_store.close(); self.result_store=None
            self.frame_cache.clear(); self.current_full_frame=None; self.latest_full_frame=None; self.active_config=None;self.prepared_metadata=None; self.prepared_ready=False
            self.prepare_btn.setEnabled(True);self.start_btn.setText("开始仿真"); self.start_btn.setEnabled(False); self.stop_btn.setEnabled(False); self.timeline.setRange(0,0); self.timeline.setValue(0); self.timeline.setEnabled(False); self.timeline_label.setText("t=0 s"); self.absolute_time_label.setText("UTC：--"); self.state_badge.setText("● 就绪"); self.status.setText("结果包已关闭")
            if self.web_ready:self.web.page().runJavaScript("resetScenario()")
            self._reset_one_click_run()
            return
        if self.worker:
            self.user_stopping = True; self.stop_btn.setEnabled(False); self.start_btn.setEnabled(False)
            self.busy_overlay.stop();self.offline_planning_started_at=None
            self.state_badge.setText("● 正在停止")
            self._sync_run_controls()
            self.worker.stop(); self.status.setText("正在结束并恢复初始状态…")
    def closeEvent(self,event):
        if self.thread is not None and self.thread.isRunning():
            event.ignore()
            if not self._closing:
                self._closing=True
                self.result_play_timer.stop()
                self.render_filter_generation+=1
                if self.render_filter_future is not None:self.render_filter_future.cancel()
                self.status.setText("正在结束仿真并关闭窗口…")
                if self.worker:self.worker.stop()
            return
        self.result_play_timer.stop()
        if self.result_store:self.result_store.close(); self.result_store=None
        self.render_filter_executor.shutdown(wait=False,cancel_futures=True)
        event.accept()


def run_app():
    from constellation_sim.ui.startup import run_app as start
    start()


if __name__ == "__main__":
    run_app()


















