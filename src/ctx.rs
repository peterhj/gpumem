use crate::{GpuLoc, GpuDev};

use cuda::runtime::{CudaDevice};

use std::cell::{Cell};

thread_local! {
  static TL_PCTX_STATE: Cell<CudaPCtxState> = Cell::new(CudaPCtxState::Rst);
}

#[derive(Clone, Copy)]
enum CudaPCtxState {
  Rst,
  Drp(GpuDev),
  Set(GpuDev),
}

pub struct CudaPCtxRef {
  dev: GpuDev,
}

impl Drop for CudaPCtxRef {
  fn drop(&mut self) {
    match TL_PCTX_STATE.with(|state| state.get()) {
      CudaPCtxState::Rst => panic!(),
      CudaPCtxState::Drp(_) => panic!(),
      CudaPCtxState::Set(prev_dev) => {
        assert_eq!(prev_dev, self.dev);
        TL_PCTX_STATE.with(|state| state.set(CudaPCtxState::Drp(self.dev)));
      }
    }
  }
}

impl GpuLoc for CudaPCtxRef {
  fn device(&self) -> GpuDev {
    self.dev
  }
}

impl CudaPCtxRef {
  pub fn set(dev: GpuDev) -> CudaPCtxRef {
    match TL_PCTX_STATE.with(|state| state.get()) {
      CudaPCtxState::Rst => {
        CudaDevice(dev.0).set_current().unwrap();
        TL_PCTX_STATE.with(|state| state.set(CudaPCtxState::Set(dev)));
      }
      CudaPCtxState::Drp(prev_dev) => {
        if prev_dev != dev {
          CudaDevice(dev.0).set_current().unwrap();
        }
        TL_PCTX_STATE.with(|state| state.set(CudaPCtxState::Set(dev)));
      }
      CudaPCtxState::Set(prev_dev) => panic!(),
    }
    CudaPCtxRef{dev}
  }
}
