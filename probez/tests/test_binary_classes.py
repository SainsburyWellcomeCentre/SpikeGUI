import unittest

from file_handling import binary_classes


class BinaryClassesTestCase(unittest.TestCase):
    """Tests binary classes"""

    def test_creation(self):
        """are the sizes of each class consistent"""
        n_channels = 384
        n_samples = 100
        base_format = 'h'
        base_format_size = 2

        data_point = binary_classes.DataPoint(base_format='h')
        time_point = binary_classes.TimePoint(data_point, n_channels)
        chunk = binary_classes.Chunk(time_point, n_samples)


        self.assertEqual(data_point.size, base_format_size)
        self.assertEqual(time_point.size, n_channels*base_format_size)
        self.assertEqual(chunk.size, n_channels*n_samples*base_format_size)


if __name__ == '__main__':
    unittest.main()