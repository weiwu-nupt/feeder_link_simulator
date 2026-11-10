"""Qt WebChannel objects exposed to the Cesium page."""

from PySide6.QtCore import QObject, Signal, Slot


class MapBridge(QObject):
    plane_filter_changed = Signal(int,int)
    viewport_changed = Signal(float,float,float,float)
    traffic_details_requested = Signal(str)
    satellite_path_requested = Signal(int,int)
    satellite_exit_comparison_requested = Signal(int,int)
    input_view_changed = Signal(str)

    @Slot(int,int)
    def setPlaneFilter(self,center_plane,plane_count):
        self.plane_filter_changed.emit(center_plane,plane_count)

    @Slot(float,float,float,float)
    def setViewport(self,west,south,east,north):
        self.viewport_changed.emit(west,south,east,north)

    @Slot(str)
    def requestTrafficDetails(self,cell_id):
        self.traffic_details_requested.emit(cell_id)

    @Slot(int,int)
    def requestSatellitePath(self,satellite_id,frame_index):
        self.satellite_path_requested.emit(satellite_id,frame_index)

    @Slot(int,int)
    def requestSatelliteComparison(self,satellite_id,frame_index):
        self.satellite_exit_comparison_requested.emit(satellite_id,frame_index)

    @Slot(str)
    def setInputView(self,mode):
        if mode in ("topology","business"):
            self.input_view_changed.emit(mode)
