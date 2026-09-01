"""Dora camera node for Hikrobot MV3D RGB-D cameras in RGB-only mode."""

from __future__ import annotations

import argparse
import ctypes as ct
import os
import pathlib
import time

import cv2
import dora
import numpy as np
import pyarrow as pa


MV3D_RGBD_OK = 0
MV3D_RGBD_MAX_IMAGE_COUNT = 10
MV3D_RGBD_MAX_ENUM_COUNT = 16
MV3D_RGBD_MAX_PATH_LENGTH = 256

DEVICE_TYPE_ETHERNET = 1 << 0
DEVICE_TYPE_USB = 1 << 1
DEVICE_TYPE_ALL = DEVICE_TYPE_ETHERNET | DEVICE_TYPE_USB

PARAM_TYPE_ENUM = 4

IMAGE_TYPE_MONO8 = 0x01080001
IMAGE_TYPE_DEPTH = 0x011000B8
IMAGE_TYPE_YUV422 = 0x02100032
IMAGE_TYPE_NV12 = 0x020C8001
IMAGE_TYPE_NV21 = 0x020C8002
IMAGE_TYPE_RGB8_PLANAR = 0x02180021
IMAGE_TYPE_JPEG = 0x80180001


class DeviceNetInfo(ct.Structure):
    _fields_ = [
        ("chMacAddress", ct.c_ubyte * 8),
        ("enIPCfgMode", ct.c_int),
        ("chCurrentIp", ct.c_char * 16),
        ("chCurrentSubNetMask", ct.c_char * 16),
        ("chDefultGateWay", ct.c_char * 16),
        ("chNetExport", ct.c_char * 16),
        ("nReserved", ct.c_ubyte * 16),
    ]


class DeviceUsbInfo(ct.Structure):
    _fields_ = [
        ("nVendorId", ct.c_uint32),
        ("nProductId", ct.c_uint32),
        ("enUsbProtocol", ct.c_int),
        ("chDeviceGUID", ct.c_char * 64),
        ("nReserved", ct.c_ubyte * 16),
    ]


class DeviceSpecialInfo(ct.Union):
    _fields_ = [
        ("stNetInfo", DeviceNetInfo),
        ("stUsbInfo", DeviceUsbInfo),
    ]


class DeviceInfo(ct.Structure):
    _fields_ = [
        ("chManufacturerName", ct.c_char * 32),
        ("chModelName", ct.c_char * 32),
        ("chDeviceVersion", ct.c_char * 32),
        ("chManufacturerSpecificInfo", ct.c_char * 48),
        ("chSerialNumber", ct.c_char * 16),
        ("chUserDefinedName", ct.c_char * 16),
        ("enDeviceType", ct.c_int),
        ("SpecialInfo", DeviceSpecialInfo),
    ]


class ImageData(ct.Structure):
    _fields_ = [
        ("enImageType", ct.c_int64),
        ("nWidth", ct.c_uint32),
        ("nHeight", ct.c_uint32),
        ("pData", ct.POINTER(ct.c_uint8)),
        ("nDataLen", ct.c_uint32),
        ("nFrameNum", ct.c_uint32),
        ("nTimeStamp", ct.c_int64),
        ("nReserved", ct.c_ubyte * 16),
    ]


class FrameData(ct.Structure):
    _fields_ = [
        ("nImageCount", ct.c_uint32),
        ("stImageData", ImageData * MV3D_RGBD_MAX_IMAGE_COUNT),
        ("nReserved", ct.c_ubyte * 16),
    ]


class EnumParam(ct.Structure):
    _fields_ = [
        ("nCurValue", ct.c_uint32),
        ("nSupportedNum", ct.c_uint32),
        ("nSupportValue", ct.c_uint32 * MV3D_RGBD_MAX_ENUM_COUNT),
    ]


class IntParam(ct.Structure):
    _fields_ = [
        ("nCurValue", ct.c_int64),
        ("nMax", ct.c_int64),
        ("nMin", ct.c_int64),
        ("nInc", ct.c_int64),
    ]


class FloatParam(ct.Structure):
    _fields_ = [
        ("fCurValue", ct.c_float),
        ("fMax", ct.c_float),
        ("fMin", ct.c_float),
    ]


class StringParam(ct.Structure):
    _fields_ = [
        ("chCurValue", ct.c_char * MV3D_RGBD_MAX_PATH_LENGTH),
    ]


class ParamInfo(ct.Union):
    _fields_ = [
        ("stIntParam", IntParam),
        ("stFloatParam", FloatParam),
        ("stEnumParam", EnumParam),
        ("stStringParam", StringParam),
        ("bBoolParam", ct.c_int32),
    ]


class Param(ct.Structure):
    _fields_ = [
        ("enParamType", ct.c_int),
        ("ParamInfo", ParamInfo),
    ]


def _decode(raw: bytes) -> str:
    return raw.split(b"\0", 1)[0].decode(errors="ignore")


def _status(status: int) -> str:
    return "OK" if status == MV3D_RGBD_OK else f"ERROR 0x{status & 0xffffffff:08x}"


def _sdk_lib_path() -> pathlib.Path:
    default_lib_dir = (
        pathlib.Path(__file__).resolve().parents[5]
        / "hikrobot"
        / "Mv3dRgbdSDK_ROS2"
        / "src"
        / "hik_rgbd"
        / "lib"
    )
    lib_dir = os.getenv(
        "HIKROBOT_MV3D_LIB",
        str(default_lib_dir),
    )
    return pathlib.Path(lib_dir) / "libMv3dRgbd.so"


def _load_sdk() -> ct.CDLL:
    lib_path = _sdk_lib_path()
    if not lib_path.exists():
        raise RuntimeError(f"Hikrobot SDK library not found: {lib_path}")

    sdk = ct.CDLL(str(lib_path))
    sdk.MV3D_RGBD_Initialize.restype = ct.c_int32
    sdk.MV3D_RGBD_Release.restype = ct.c_int32
    sdk.MV3D_RGBD_GetDeviceNumber.argtypes = [ct.c_uint32, ct.POINTER(ct.c_uint32)]
    sdk.MV3D_RGBD_GetDeviceNumber.restype = ct.c_int32
    sdk.MV3D_RGBD_GetDeviceList.argtypes = [
        ct.c_uint32,
        ct.POINTER(DeviceInfo),
        ct.c_uint32,
        ct.POINTER(ct.c_uint32),
    ]
    sdk.MV3D_RGBD_GetDeviceList.restype = ct.c_int32
    sdk.MV3D_RGBD_OpenDevice.argtypes = [ct.POINTER(ct.c_void_p), ct.POINTER(DeviceInfo)]
    sdk.MV3D_RGBD_OpenDevice.restype = ct.c_int32
    sdk.MV3D_RGBD_OpenDeviceBySerialNumber.argtypes = [
        ct.POINTER(ct.c_void_p),
        ct.c_char_p,
    ]
    sdk.MV3D_RGBD_OpenDeviceBySerialNumber.restype = ct.c_int32
    sdk.MV3D_RGBD_CloseDevice.argtypes = [ct.POINTER(ct.c_void_p)]
    sdk.MV3D_RGBD_CloseDevice.restype = ct.c_int32
    sdk.MV3D_RGBD_Start.argtypes = [ct.c_void_p]
    sdk.MV3D_RGBD_Start.restype = ct.c_int32
    sdk.MV3D_RGBD_Stop.argtypes = [ct.c_void_p]
    sdk.MV3D_RGBD_Stop.restype = ct.c_int32
    sdk.MV3D_RGBD_FetchFrame.argtypes = [ct.c_void_p, ct.POINTER(FrameData), ct.c_uint32]
    sdk.MV3D_RGBD_FetchFrame.restype = ct.c_int32
    sdk.MV3D_RGBD_SetParam.argtypes = [ct.c_void_p, ct.c_char_p, ct.POINTER(Param)]
    sdk.MV3D_RGBD_SetParam.restype = ct.c_int32
    return sdk


def _check(label: str, status: int) -> None:
    if status != MV3D_RGBD_OK:
        raise RuntimeError(f"{label}: {_status(status)}")


def _set_enum(sdk: ct.CDLL, handle: ct.c_void_p, key: str, value: int) -> None:
    param = Param()
    param.enParamType = PARAM_TYPE_ENUM
    param.ParamInfo.stEnumParam.nCurValue = value
    _check(f"Set {key}={value}", sdk.MV3D_RGBD_SetParam(handle, key.encode(), ct.byref(param)))


def _image_bytes(image: ImageData) -> np.ndarray:
    return np.ctypeslib.as_array(image.pData, shape=(int(image.nDataLen),))


def _image_to_bgr(image: ImageData) -> np.ndarray | None:
    data = _image_bytes(image)
    width = int(image.nWidth)
    height = int(image.nHeight)
    image_type = int(image.enImageType)

    if image_type == IMAGE_TYPE_YUV422:
        yuv = data.reshape((height, width, 2))
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_YUY2)
    if image_type == IMAGE_TYPE_NV12:
        yuv = data.reshape((height + height // 2, width))
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
    if image_type == IMAGE_TYPE_NV21:
        yuv = data.reshape((height + height // 2, width))
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV21)
    if image_type == IMAGE_TYPE_RGB8_PLANAR:
        pixels = width * height
        r = data[:pixels].reshape((height, width))
        g = data[pixels : pixels * 2].reshape((height, width))
        b = data[pixels * 2 : pixels * 3].reshape((height, width))
        return cv2.merge((b, g, r))
    if image_type == IMAGE_TYPE_JPEG:
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image_type == IMAGE_TYPE_MONO8:
        mono = data.reshape((height, width))
        return cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)
    return None


def _select_rgb_image(frame: FrameData) -> ImageData | None:
    preferred = {
        IMAGE_TYPE_YUV422,
        IMAGE_TYPE_NV12,
        IMAGE_TYPE_NV21,
        IMAGE_TYPE_RGB8_PLANAR,
        IMAGE_TYPE_JPEG,
    }
    fallback = None
    for index in range(int(frame.nImageCount)):
        image = frame.stImageData[index]
        if int(image.enImageType) in preferred:
            return image
        if fallback is None and int(image.enImageType) == IMAGE_TYPE_MONO8:
            fallback = image
    return fallback


class HikrobotCamera:
    def __init__(self, device_index: int, serial_number: str | None, working_mode: int, image_mode: int):
        self.sdk = _load_sdk()
        self.handle = ct.c_void_p()
        self.started = False
        _check("Initialize", self.sdk.MV3D_RGBD_Initialize())

        if serial_number:
            _check(
                f"OpenDeviceBySerialNumber({serial_number})",
                self.sdk.MV3D_RGBD_OpenDeviceBySerialNumber(
                    ct.byref(self.handle),
                    serial_number.encode(),
                ),
            )
            print(f"[hikrobot] opened serial={serial_number}")
        else:
            count = ct.c_uint32()
            _check(
                "GetDeviceNumber",
                self.sdk.MV3D_RGBD_GetDeviceNumber(DEVICE_TYPE_ALL, ct.byref(count)),
            )
            if count.value <= 0:
                raise RuntimeError("No Hikrobot MV3D RGB-D camera found.")
            if device_index < 0 or device_index >= count.value:
                raise RuntimeError(f"DEVICE_INDEX={device_index} out of range, count={count.value}")

            devices = (DeviceInfo * count.value)()
            listed = ct.c_uint32()
            _check(
                "GetDeviceList",
                self.sdk.MV3D_RGBD_GetDeviceList(
                    DEVICE_TYPE_ALL,
                    devices,
                    count.value,
                    ct.byref(listed),
                ),
            )
            if listed.value <= 0:
                raise RuntimeError("Hikrobot SDK listed zero devices.")
            device = devices[device_index]
            _check("OpenDevice", self.sdk.MV3D_RGBD_OpenDevice(ct.byref(self.handle), ct.byref(device)))
            print(
                "[hikrobot] opened "
                f"index={device_index} model={_decode(device.chModelName)} "
                f"serial={_decode(device.chSerialNumber)}"
            )

        _set_enum(self.sdk, self.handle, "CameraWorkingMode", working_mode)
        _set_enum(self.sdk, self.handle, "ImageMode", image_mode)
        _check("Start", self.sdk.MV3D_RGBD_Start(self.handle))
        self.started = True
        print(f"[hikrobot] RGB mode started working_mode={working_mode} image_mode={image_mode}")

    def fetch_bgr(self, timeout_ms: int) -> np.ndarray | None:
        frame = FrameData()
        status = self.sdk.MV3D_RGBD_FetchFrame(self.handle, ct.byref(frame), timeout_ms)
        if status != MV3D_RGBD_OK:
            print(f"[hikrobot] FetchFrame: {_status(status)}")
            return None

        image = _select_rgb_image(frame)
        if image is None:
            print(f"[hikrobot] no RGB image in frame, image_count={frame.nImageCount}")
            return None
        bgr = _image_to_bgr(image)
        if bgr is None or bgr.size == 0:
            print(f"[hikrobot] unsupported image type: 0x{int(image.enImageType):08x}")
            return None
        return bgr

    def close(self) -> None:
        if self.started:
            self.sdk.MV3D_RGBD_Stop(self.handle)
            self.started = False
        if self.handle:
            self.sdk.MV3D_RGBD_CloseDevice(ct.byref(self.handle))
        self.sdk.MV3D_RGBD_Release()


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None or value == "" else int(value)


def _env_optional_int(name: str) -> int | None:
    value = os.getenv(name)
    return None if value is None or value == "" else int(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hikrobot MV3D RGB Dora camera node")
    parser.add_argument("--name", default="hikrobot-rgb-camera")
    parser.add_argument("--device-index", type=int, default=_env_int("DEVICE_INDEX", 0))
    parser.add_argument("--serial-number", default=os.getenv("SERIAL_NUMBER"))
    parser.add_argument("--image-width", type=int, default=_env_optional_int("IMAGE_WIDTH"))
    parser.add_argument("--image-height", type=int, default=_env_optional_int("IMAGE_HEIGHT"))
    parser.add_argument("--jpeg-quality", type=int, default=_env_int("JPEG_QUALITY", 90))
    parser.add_argument("--fetch-timeout-ms", type=int, default=_env_int("FETCH_TIMEOUT_MS", 1000))
    parser.add_argument("--working-mode", type=int, default=_env_int("WORKING_MODE", 2))
    parser.add_argument("--image-mode", type=int, default=_env_int("IMAGE_MODE", 8))
    args = parser.parse_args()

    node = dora.Node(args.name)
    camera = HikrobotCamera(
        device_index=args.device_index,
        serial_number=args.serial_number,
        working_mode=args.working_mode,
        image_mode=args.image_mode,
    )
    try:
        for event in node:
            if event["type"] == "STOP":
                break
            if event["type"] == "INPUT_CLOSED" and event.get("id") == "tick":
                break
            if event["type"] != "INPUT" or event["id"] != "tick":
                continue

            frame = camera.fetch_bgr(args.fetch_timeout_ms)
            if frame is None:
                continue
            capture_timestamp_ns = time.time_ns()

            if args.image_width and args.image_height:
                if frame.shape[1] != args.image_width or frame.shape[0] != args.image_height:
                    frame = cv2.resize(
                        frame,
                        (args.image_width, args.image_height),
                        interpolation=cv2.INTER_AREA,
                    )

            encode_params = [cv2.IMWRITE_JPEG_QUALITY, int(args.jpeg_quality)]
            ok, encoded = cv2.imencode(".jpeg", frame, encode_params)
            if not ok:
                print("[hikrobot] JPEG encode failed")
                continue

            metadata = dict(event["metadata"])
            metadata.pop("timestamp", None)
            metadata["capture_timestamp_ns"] = capture_timestamp_ns
            metadata["encoding"] = "jpeg"
            metadata["width"] = int(frame.shape[1])
            metadata["height"] = int(frame.shape[0])
            metadata["primitive"] = "image"
            node.send_output("image", pa.array(encoded.ravel()), metadata)
    finally:
        camera.close()


if __name__ == "__main__":
    main()
