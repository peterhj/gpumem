#![feature(align_offset)]
#![feature(optin_builtin_traits)]

extern crate cuda;
extern crate podmem;

pub use crate::ctx::{CudaPCtxRef};

use cuda::runtime::{
  CudaStream, CudaMemcpyKind,
  cuda_alloc_host_with_flags, cuda_free_host,
  cuda_alloc_device, cuda_free_device,
  cuda_memset_async, cuda_memcpy_async,
};
use podmem::{DmaRegion, DmaRegionMut, ZeroBits};

use std::mem::{align_of, size_of};
use std::ops::{Deref, RangeBounds, Bound};
use std::slice::{from_raw_parts, from_raw_parts_mut};

pub mod ctx;

pub trait GpuLoc {
  fn device(&self) -> GpuDev;
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct GpuDev(pub i32);

impl From<i32> for GpuDev {
  fn from(dev: i32) -> GpuDev {
    GpuDev(dev)
  }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum GpuDom {
  Host,
  Dev(GpuDev),
}

pub trait GpuDelay {
  //type Data: Copy + 'static;
}

// FIXME: implement `GpuDelay` for `Copy` types?

impl GpuDelay for () {
  //type Data = ();
}

/*pub trait GpuDelayed<V: GpuDelay>: Deref<Target=V> {
  //fn domain(&self) -> GpuDom;
  //fn delayed_ptr(&self) -> *const <V as GpuDelay>::Data;
}

pub trait GpuDelayedMut<V: GpuDelay>: GpuDelayed<V> {
  //fn delayed_ptr_mut(&self) -> *mut <V as GpuDelay>::Data;
}*/

/*pub trait GpuRegion<T: Copy + 'static> {
  fn device(&self) -> GpuDev;
  fn as_devptr(&self) -> *const T;
  fn region_len(&self) -> usize;
}*/

pub struct GpuPinnedMem<T: Copy + 'static> {
  dev:  GpuDev,
  ptr:  *mut T,
  len:  usize,
}

impl<T: Copy + 'static> GpuDelay for GpuPinnedMem<T> {
  //type Data = T;
}

impl<T: Copy + 'static> Drop for GpuPinnedMem<T> {
  fn drop(&mut self) {
    // TODO: synchronization?
    unsafe {
      match cuda_free_host(self.ptr as *mut u8) {
        Ok(_) => {}
        Err(e) => panic!("cudaFreeHost failed: {:?} ({})", e, e.get_string()),
      }
    }
  }
}

impl<T: Copy + 'static> GpuPinnedMem<T> {
  pub unsafe fn alloc(len: usize, dev: GpuDev) -> GpuPinnedMem<T> {
    GpuPinnedMem::alloc_with_flags(len, dev, 0)
  }

  pub unsafe fn alloc_with_flags(len: usize, dev: GpuDev, flags: u32) -> GpuPinnedMem<T> {
    let size = len * size_of::<T>();
    let raw_ptr: *mut u8 = {
      let _ctx = CudaPCtxRef::set(dev);
      // TODO: synchronization?
      match cuda_alloc_host_with_flags(size, flags) {
        Ok(ptr) => ptr,
        Err(e) => panic!("cudaHostAlloc failed: {:?} ({})", e, e.get_string()),
      }
    };
    // NB: Non-aligned accesses are sort of supported, but can be totally broken
    // for some word sizes (namely, 8 and 16 bytes). Easier to enforce alignment
    // at all stages, including here during allocation.
    //
    // Also see section 5.3.2 of the CUDA C programming guide:
    // <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses>.
    let ptr: *mut T = raw_ptr as *mut T;
    match (size_of::<T>() % align_of::<T>(), ptr.align_offset(align_of::<T>())) {
      (0, 0) => {}
      (0, _) => panic!("cudaHostAlloc returned non-naturally aligned pointer"),
      (_, _) => panic!("size is not a multiple of alignment"),
    }
    GpuPinnedMem{
      dev,
      ptr,
      len,
    }
  }

  pub fn len(&self) -> usize {
    self.len
  }

  pub fn as_ptr(&self) -> *const T {
    self.ptr
  }

  pub fn as_mut_ptr(&self) -> *mut T {
    self.ptr
  }

  pub unsafe fn as_slice(&self) -> &[T] {
    from_raw_parts(self.ptr, self.len)
  }

  pub unsafe fn as_mut_slice(&mut self) -> &mut [T] {
    from_raw_parts_mut(self.ptr, self.len)
  }
}

pub struct GpuVMem<T: Copy + 'static> {
  dev:  GpuDev,
  dptr: *mut T,
  len:  usize,
}

impl<T: Copy + 'static> GpuDelay for GpuVMem<T> {
  //type Data = T;
}

impl<T: Copy + 'static> GpuLoc for GpuVMem<T> {
  fn device(&self) -> GpuDev {
    self.dev
  }
}

impl<T: Copy + 'static> DmaRegion<T> for GpuVMem<T> {
  fn dma_region_len(&self) -> usize {
    self.len
  }

  fn as_dma_ptr(&self) -> *const T {
    self.dptr
  }
}

impl<T: Copy + 'static> DmaRegionMut<T> for GpuVMem<T> {
  fn as_dma_ptr_mut(&self) -> *mut T {
    self.dptr
  }
}

impl<T: Copy + 'static> Drop for GpuVMem<T> {
  fn drop(&mut self) {
    // NB: This _will_ implicitly synchronize; see:
    // <https://github.com/thrust/thrust/issues/905>
    // <https://cs.unc.edu/~tamert/papers/ecrts18b.pdf>
    unsafe {
      match cuda_free_device(self.dptr as *mut u8) {
        Ok(_) => {}
        Err(e) => panic!("cudaFree failed: {:?} ({})", e, e.get_string()),
      }
    }
  }
}

impl<T: ZeroBits + 'static> GpuVMem<T> {
  pub fn zeroed(len: usize, dev: GpuDev, cuda_stream: &mut CudaStream) -> GpuVMem<T> {
    let mut mem: GpuVMem<T> = unsafe { GpuVMem::alloc(len, dev) };
    match unsafe { cuda_memset_async(
        mem.dptr as *mut u8,
        0,
        mem.len() * size_of::<T>(),
        cuda_stream,
    ) } {
      Err(_) => panic!(),
      Ok(_) => {}
    }
    mem
  }
}

impl<T: Copy + 'static> GpuVMem<T> {
  pub unsafe fn alloc(len: usize, dev: GpuDev) -> GpuVMem<T> {
    let size = len * size_of::<T>();
    let raw_dptr: *mut u8 = {
      let _ctx = CudaPCtxRef::set(dev);
      // NB: This _may_ implicitly synchronize; see:
      // <https://github.com/thrust/thrust/issues/905>
      match cuda_alloc_device(size) {
        Ok(dptr) => dptr,
        Err(e) => panic!("cudaMalloc failed: {:?} ({})", e, e.get_string()),
      }
    };
    // NB: Non-aligned accesses are sort of supported, but can be totally broken
    // for some word sizes (namely, 8 and 16 bytes). Easier to enforce alignment
    // at all stages, including here during allocation.
    //
    // Also see section 5.3.2 of the CUDA C programming guide:
    // <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses>.
    let dptr: *mut T = raw_dptr as *mut T;
    match (size_of::<T>() % align_of::<T>(), dptr.align_offset(align_of::<T>())) {
      (0, 0) => {}
      (0, _) => panic!("cudaMalloc returned non-naturally aligned pointer"),
      (_, _) => panic!("size is not a multiple of alignment"),
    }
    GpuVMem{
      dev,
      dptr,
      len,
    }
  }

  pub fn len(&self) -> usize {
    self.len
  }

  pub fn as_dptr(&self) -> *const T {
    self.dptr
  }

  pub fn as_mut_dptr(&self) -> *mut T {
    self.dptr
  }

  pub fn copy_from_slice_sync(&self, src: &[T], cuda_stream: &mut CudaStream) {
    assert_eq!(self.len(), src.len());
    /*match cuda_stream.synchronize() {
      Err(e) => panic!("copy_from_slice_sync: cudaStreamSynchronize: {:?}", e),
      Ok(_) => {}
    }*/
    match unsafe { cuda_memcpy_async(
        self.as_dma_ptr_mut(),
        src.as_ptr(),
        src.len(),
        CudaMemcpyKind::HostToDevice,
        cuda_stream,
    ) } {
      Err(e) => panic!("copy_from_slice_sync: cudaMemcpyAsync: {:?}", e),
      Ok(_) => {}
    }
    match cuda_stream.synchronize() {
      Err(e) => panic!("copy_from_slice_sync: cudaStreamSynchronize: {:?}", e),
      Ok(_) => {}
    }
  }

  pub fn copy_to_slice_sync(&self, dst: &mut [T], cuda_stream: &mut CudaStream) {
    assert_eq!(self.len(), dst.len());
    /*match cuda_stream.synchronize() {
      Err(e) => panic!("copy_to_slice_sync: cudaStreamSynchronize: {:?}", e),
      Ok(_) => {}
    }*/
    match unsafe { cuda_memcpy_async(
        dst.as_mut_ptr(),
        self.as_dma_ptr(),
        dst.len(),
        CudaMemcpyKind::DeviceToHost,
        cuda_stream,
    ) } {
      Err(e) => panic!("copy_to_slice_sync: cudaMemcpyAsync: {:?}", e),
      Ok(_) => {}
    }
    match cuda_stream.synchronize() {
      Err(e) => panic!("copy_to_slice_sync: cudaStreamSynchronize: {:?}", e),
      Ok(_) => {}
    }
  }

  pub fn copy_to_host_buf_sync(&self, dst_ptr: *mut T, dst_len: usize, cuda_stream: &mut CudaStream) {
    assert_eq!(self.len(), dst_len);
    /*match cuda_stream.synchronize() {
      Err(e) => panic!("copy_to_slice_sync: cudaStreamSynchronize: {:?}", e),
      Ok(_) => {}
    }*/
    match unsafe { cuda_memcpy_async(
        dst_ptr,
        self.as_dma_ptr(),
        dst_len,
        CudaMemcpyKind::DeviceToHost,
        cuda_stream,
    ) } {
      Err(e) => panic!("copy_to_slice_sync: cudaMemcpyAsync: {:?}", e),
      Ok(_) => {}
    }
    match cuda_stream.synchronize() {
      Err(e) => panic!("copy_to_slice_sync: cudaStreamSynchronize: {:?}", e),
      Ok(_) => {}
    }
  }

  pub fn slice<R: RangeBounds<usize>>(&self, range: R) -> GpuUnsafeSlice<T> {
    let start = match range.start_bound() {
      Bound::Included(idx) => *idx,
      Bound::Excluded(idx) => *idx + 1,
      Bound::Unbounded => 0,
    };
    let end = match range.end_bound() {
      Bound::Included(idx) => *idx + 1,
      Bound::Excluded(idx) => *idx,
      Bound::Unbounded => self.len,
    };
    assert!(start <= end);
    assert!(end <= self.len);
    let slice_dptr = unsafe { self.dptr.offset(start as isize) };
    GpuUnsafeSlice{
      dev:  self.dev,
      dptr: slice_dptr,
      len:  end - start,
    }
  }
}

pub struct GpuUnsafeSlice<T: Copy + 'static> {
  dev:  GpuDev,
  dptr: *mut T,
  len:  usize,
}

impl<T: Copy + 'static> GpuLoc for GpuUnsafeSlice<T> {
  fn device(&self) -> GpuDev {
    self.dev
  }
}

impl<T: Copy + 'static> DmaRegion<T> for GpuUnsafeSlice<T> {
  fn dma_region_len(&self) -> usize {
    self.len
  }

  fn as_dma_ptr(&self) -> *const T {
    self.dptr
  }
}

impl<T: Copy + 'static> DmaRegionMut<T> for GpuUnsafeSlice<T> {
  fn as_dma_ptr_mut(&self) -> *mut T {
    self.dptr
  }
}
