import unittest
from unittest.mock import patch, MagicMock
import sys

# Fully mock the tensorflow hierarchy
mock_tf = MagicMock()
sys.modules['tensorflow'] = mock_tf
sys.modules['tensorflow.keras'] = mock_tf.keras
sys.modules['tensorflow.keras.models'] = mock_tf.keras.models
sys.modules['tensorflow.keras.layers'] = mock_tf.keras.layers
sys.modules['tensorflow.keras.optimizers'] = mock_tf.keras.optimizers

# Mock other dependencies
sys.modules['cv2'] = MagicMock()
sys.modules['matplotlib'] = MagicMock()
sys.modules['matplotlib.pyplot'] = MagicMock()
sys.modules['tqdm'] = MagicMock()
sys.modules['numpy'] = MagicMock()

# Now import load_model from gan.src.inference
from gan.src.inference import load_model

class TestSecurityFix(unittest.TestCase):
    def test_load_model_uses_safe_mode(self):
        model_path = "test_model.h5"
        load_model(model_path)

        # Verify that tf.keras.models.load_model was called correctly
        mock_tf.keras.models.load_model.assert_called_with(
            model_path, compile=False, safe_mode=True
        )

if __name__ == '__main__':
    unittest.main()
