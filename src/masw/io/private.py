"""Files their owner alone may read or write, on every system, for secrets such as an API key: on
Linux and macOS a file of mode 0o600; on Windows a file whose access list grants the current user
alone, its folder's rights not inherited. Standard library only."""

import os
import sys
from pathlib import Path


def write_private(path: Path, text: str) -> None:
    """`text` in `path`, whole or not at all: written to a file beside it, made private when it is
    created (before anything is written in it), then renamed over `path`."""
    partial = path.with_name(f".{path.name}.new")
    partial.unlink(missing_ok=True)
    try:
        with os.fdopen(_create_private(partial), "w", encoding="utf-8") as file:
            file.write(text)
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


def make_private(path: Path) -> None:
    """`path`, an existing file, made its owner's alone, as write_private makes its files."""
    _protect(path)


def is_private(path: Path) -> bool:
    """Whether the owner of `path` alone may read or write it."""
    return _is_private(path)


if sys.platform == "win32":
    import ctypes
    import msvcrt
    from ctypes import wintypes

    _advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    _TOKEN_QUERY = 0x0008
    _TOKEN_USER = 1  # TOKEN_INFORMATION_CLASS: the token's user
    _SDDL_REVISION_1 = 1
    _SE_FILE_OBJECT = 1
    _DACL_SECURITY_INFORMATION = 0x00000004
    _PROTECTED_DACL_SECURITY_INFORMATION = 0x80000000
    _GENERIC_WRITE = 0x40000000
    _CREATE_NEW = 1
    _FILE_ATTRIBUTE_NORMAL = 0x80
    _INVALID_HANDLE_VALUE = wintypes.HANDLE(-1).value

    class _SecurityAttributes(ctypes.Structure):
        _fields_ = [
            ("nLength", wintypes.DWORD),
            ("lpSecurityDescriptor", wintypes.LPVOID),
            ("bInheritHandle", wintypes.BOOL),
        ]

    _kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    _kernel32.GetCurrentProcess.argtypes = []
    _kernel32.CloseHandle.restype = wintypes.BOOL
    _kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    _kernel32.LocalFree.restype = wintypes.LPVOID
    _kernel32.LocalFree.argtypes = [wintypes.LPVOID]
    _kernel32.CreateFileW.restype = wintypes.HANDLE
    _kernel32.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(_SecurityAttributes),
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    _advapi32.OpenProcessToken.restype = wintypes.BOOL
    _advapi32.OpenProcessToken.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.HANDLE),
    ]
    _advapi32.GetTokenInformation.restype = wintypes.BOOL
    _advapi32.GetTokenInformation.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    _advapi32.ConvertSidToStringSidW.restype = wintypes.BOOL
    _advapi32.ConvertSidToStringSidW.argtypes = [wintypes.LPVOID, ctypes.POINTER(wintypes.LPWSTR)]
    _advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW.restype = wintypes.BOOL
    _advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.LPVOID),
        ctypes.POINTER(wintypes.ULONG),
    ]
    _advapi32.GetSecurityDescriptorDacl.restype = wintypes.BOOL
    _advapi32.GetSecurityDescriptorDacl.argtypes = [
        wintypes.LPVOID,
        ctypes.POINTER(wintypes.BOOL),
        ctypes.POINTER(wintypes.LPVOID),
        ctypes.POINTER(wintypes.BOOL),
    ]
    _advapi32.SetNamedSecurityInfoW.restype = wintypes.DWORD
    _advapi32.SetNamedSecurityInfoW.argtypes = [
        wintypes.LPWSTR,
        ctypes.c_int,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.LPVOID,
        wintypes.LPVOID,
        wintypes.LPVOID,
    ]
    _advapi32.GetNamedSecurityInfoW.restype = wintypes.DWORD
    _advapi32.GetNamedSecurityInfoW.argtypes = [
        wintypes.LPCWSTR,
        ctypes.c_int,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.LPVOID),
        ctypes.POINTER(wintypes.LPVOID),
        ctypes.POINTER(wintypes.LPVOID),
        ctypes.POINTER(wintypes.LPVOID),
        ctypes.POINTER(wintypes.LPVOID),
    ]
    _advapi32.ConvertSecurityDescriptorToStringSecurityDescriptorW.restype = wintypes.BOOL
    _advapi32.ConvertSecurityDescriptorToStringSecurityDescriptorW.argtypes = [
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.LPWSTR),
        ctypes.POINTER(wintypes.ULONG),
    ]

    def _failed() -> OSError:
        return ctypes.WinError(ctypes.get_last_error())

    def _user_sid() -> str:
        """The current user's security identifier, as SDDL writes it (S-1-5-21-...)."""
        token = wintypes.HANDLE()
        if not _advapi32.OpenProcessToken(
            _kernel32.GetCurrentProcess(), _TOKEN_QUERY, ctypes.byref(token)
        ):
            raise _failed()
        try:
            size = wintypes.DWORD()
            _advapi32.GetTokenInformation(token, _TOKEN_USER, None, 0, ctypes.byref(size))
            buffer = ctypes.create_string_buffer(size.value)
            if not _advapi32.GetTokenInformation(
                token, _TOKEN_USER, buffer, size, ctypes.byref(size)
            ):
                raise _failed()
            # TOKEN_USER starts with a SID_AND_ATTRIBUTES, which starts with the SID's address.
            sid = wintypes.LPVOID.from_buffer(buffer).value
            text = wintypes.LPWSTR()
            if not _advapi32.ConvertSidToStringSidW(sid, ctypes.byref(text)):
                raise _failed()
            try:
                return str(text.value)
            finally:
                _kernel32.LocalFree(ctypes.cast(text, wintypes.LPVOID))
        finally:
            _kernel32.CloseHandle(token)

    def _owner_alone() -> str:
        """The access list of a private file: the current user allowed everything, nobody else
        anything, the folder's rights not inherited (P)."""
        return f"D:P(A;;FA;;;{_user_sid()})"

    def _descriptor() -> wintypes.LPVOID:
        """A security descriptor holding _owner_alone's list, for LocalFree to free."""
        descriptor = wintypes.LPVOID()
        if not _advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW(
            _owner_alone(), _SDDL_REVISION_1, ctypes.byref(descriptor), None
        ):
            raise _failed()
        return descriptor

    def _create_private(path: Path) -> int:
        """`path` created new, with _owner_alone's list from the start, open for writing."""
        descriptor = _descriptor()
        try:
            attributes = _SecurityAttributes(ctypes.sizeof(_SecurityAttributes), descriptor, False)
            handle = _kernel32.CreateFileW(
                str(path),
                _GENERIC_WRITE,
                0,
                ctypes.byref(attributes),
                _CREATE_NEW,
                _FILE_ATTRIBUTE_NORMAL,
                None,
            )
            if handle in (None, _INVALID_HANDLE_VALUE):
                raise _failed()
        finally:
            _kernel32.LocalFree(descriptor)
        return msvcrt.open_osfhandle(handle, os.O_WRONLY)

    def _protect(path: Path) -> None:
        descriptor = _descriptor()
        try:
            present, dacl, defaulted = wintypes.BOOL(), wintypes.LPVOID(), wintypes.BOOL()
            if not _advapi32.GetSecurityDescriptorDacl(
                descriptor, ctypes.byref(present), ctypes.byref(dacl), ctypes.byref(defaulted)
            ):
                raise _failed()
            error = _advapi32.SetNamedSecurityInfoW(
                str(path),
                _SE_FILE_OBJECT,
                _DACL_SECURITY_INFORMATION | _PROTECTED_DACL_SECURITY_INFORMATION,
                None,
                None,
                dacl,
                None,
            )
            if error:
                raise ctypes.WinError(error)
        finally:
            _kernel32.LocalFree(descriptor)

    def _is_private(path: Path) -> bool:
        descriptor = wintypes.LPVOID()
        error = _advapi32.GetNamedSecurityInfoW(
            str(path),
            _SE_FILE_OBJECT,
            _DACL_SECURITY_INFORMATION,
            None,
            None,
            None,
            None,
            ctypes.byref(descriptor),
        )
        if error:
            raise ctypes.WinError(error)
        try:
            text = wintypes.LPWSTR()
            if not _advapi32.ConvertSecurityDescriptorToStringSecurityDescriptorW(
                descriptor, _SDDL_REVISION_1, _DACL_SECURITY_INFORMATION, ctypes.byref(text), None
            ):
                raise _failed()
            try:
                found = str(text.value)
            finally:
                _kernel32.LocalFree(ctypes.cast(text, wintypes.LPVOID))
        finally:
            _kernel32.LocalFree(descriptor)
        # "D:" and the list's flags (P: protected, AI: inherited automatically), then its entries.
        flags, _, entries = found.removeprefix("D:").partition("(")
        expected = _owner_alone().removeprefix("D:P")
        return "P" in flags and f"({entries}" == expected

else:

    def _create_private(path: Path) -> int:
        """`path` created new, of mode 0o600 whatever the umask, open for writing."""
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.fchmod(descriptor, 0o600)
        return descriptor

    def _protect(path: Path) -> None:
        path.chmod(0o600)

    def _is_private(path: Path) -> bool:
        return path.stat().st_mode & 0o777 == 0o600
