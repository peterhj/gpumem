use crate::{GpuDev};

use cudart::{CudaDevice};

use std::cell::{Cell, RefCell};

thread_local! {
  static ROOT_DEVICE:   Cell<Option<i32>> = Cell::new(None);
  static DEVICE_STACK:  RefCell<Vec<i32>> = RefCell::new(Vec::new());
}

pub struct GpuCtxGuard {
  dev:  i32,
  pop:  i32,
}

impl !Send for GpuCtxGuard {}
impl !Sync for GpuCtxGuard {}

impl Drop for GpuCtxGuard {
  fn drop(&mut self) {
    DEVICE_STACK.with(|dev_stack| {
      let mut dev_stack = dev_stack.borrow_mut();
      match dev_stack.pop() {
        None => panic!("bug"),
        Some(d) => assert_eq!(d, self.dev),
      }
      match CudaDevice(self.pop).set_current() {
        Err(e) => {
          panic!("set current device failed: {:?} ({})", e, e.get_string());
        }
        Ok(_) => {}
      }
    });
  }
}

impl GpuCtxGuard {
  pub fn new(dev: GpuDev) -> GpuCtxGuard {
    DEVICE_STACK.with(|dev_stack| {
      ROOT_DEVICE.with(|root_dev| {
        let mut dev_stack = dev_stack.borrow_mut();
        let depth = dev_stack.len();
        let pop = match depth {
          0 => {
            match root_dev.get() {
              None => {
                let curr_dev = match CudaDevice::get_current() {
                  Err(e) => {
                    panic!("get current device failed: {:?} ({})", e, e.get_string());
                  }
                  Ok(device) => {
                    device.0
                  }
                };
                root_dev.set(Some(curr_dev));
                curr_dev
              }
              Some(d) => d,
            }
          }
          _ => {
            dev_stack[depth - 1]
          }
        };
        match CudaDevice(dev.0).set_current() {
          Err(e) => {
            panic!("set current device failed: {:?} ({})", e, e.get_string());
          }
          Ok(_) => {}
        }
        dev_stack.push(dev.0);
        GpuCtxGuard{dev: dev.0, pop}
      })
    })
  }
}
