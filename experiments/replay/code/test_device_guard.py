"""Hardware identity formatting regression, independent of model calculations."""
import unittest
from unittest.mock import patch
from replay import authorize_gpu

class DeviceGuardTest(unittest.TestCase):
    def setUp(self):
        guard=patch.dict("os.environ", {"GRPO_REPLAY_GPU_UUID":"538481fa-2691-f11d-b79d-666a837587d7"})
        guard.start(); self.addCleanup(guard.stop)
    def test_accepted_representations(self):
        uuid='538481fa-2691-f11d-b79d-666a837587d7'
        for value in [uuid,'GPU-'+uuid]:
            authorize_gpu(dict(gpu_uuid=value,visible_devices='GPU-'+uuid))

    def test_other_device_rejected(self):
        with self.assertRaises(ValueError):
            authorize_gpu(dict(gpu_uuid='another-device',visible_devices='GPU-538481fa-2691-f11d-b79d-666a837587d7'))

    def test_extra_visible_device_rejected(self):
        with self.assertRaises(ValueError):
            authorize_gpu(dict(gpu_uuid='538481fa-2691-f11d-b79d-666a837587d7',visible_devices='0,1'))

if __name__=='__main__':unittest.main()
