import torch

class GridGenerator:
    def __init__(self, height: int, width: int, kernel_size, stride=1):
        self.height = height
        self.width = width
        self.kernel_size = kernel_size  # (Kh, Kw)
        self.stride = stride  # (H, W)

    def createSamplingPattern(self):
        """
        :return: (1, H*Kh, W*Kw, (Lat, Lon)) sampling pattern
        """
        kerX, kerY = self.createKernel()  # (Kh, Kw)

        # create some values using in generating lat/lon sampling pattern
        rho = torch.sqrt(kerX ** 2 + kerY ** 2)
        Kh, Kw = self.kernel_size
        # when the value of rho at center is zero, some lat values explode to `nan`.
        if Kh % 2 and Kw % 2:
            rho[Kh // 2][Kw // 2] = 1e-8

        nu = torch.atan(rho)
        cos_nu = torch.cos(nu)
        sin_nu = torch.sin(nu)

        stride_h, stride_w = self.stride
        h_range = torch.arange(0, self.height, stride_h, dtype=torch.float32)
        w_range = torch.arange(0, self.width, stride_w, dtype=torch.float32)

        lat_range = ((h_range / self.height) - 0.5) * torch.pi
        lon_range = ((w_range / self.width) - 0.5) * (2 * torch.pi)

        # generate latitude sampling pattern
        lat = torch.stack([
            torch.asin(cos_nu * torch.sin(_lat) + kerY * sin_nu * torch.cos(_lat) / rho)
            for _lat in lat_range
        ])  # (H, Kh, Kw)

        lat = torch.stack([lat for _ in lon_range])  # (W, H, Kh, Kw)
        lat = lat.permute(1, 0, 2, 3)  # (H, W, Kh, Kw)

        # generate longitude sampling pattern
        lon = torch.stack([
            torch.atan(kerX * sin_nu / (rho * torch.cos(_lat) * cos_nu - kerY * torch.sin(_lat) * sin_nu))
            for _lat in lat_range
        ])  # (H, Kh, Kw)

        lon = torch.stack([lon + _lon for _lon in lon_range])  # (W, H, Kh, Kw)
        lon = lon.permute(1, 0, 2, 3)  # (H, W, Kh, Kw)

        # (radian) -> (index of pixel)
        lat = (lat / torch.pi + 0.5) * self.height
        lon = ((lon / (2 * torch.pi) + 0.5) * self.width) % self.width

        LatLon = torch.stack((lat, lon), dim=-1)  # (H, W, Kh, Kw, 2)
        LatLon = LatLon.permute(0, 2, 1, 3, 4)  # (H, Kh, W, Kw, 2)

        H, Kh, W, Kw, d = LatLon.shape
        LatLon = LatLon.reshape(1, H * Kh, W * Kw, d)  # (1, H*Kh, W*Kw, 2)

        return LatLon

    def createKernel(self):
        """
        :return: (Ky, Kx) kernel pattern
        """
        Kh, Kw = self.kernel_size

        delta_lat = torch.pi / self.height
        delta_lon = 2 * torch.pi / self.width

        range_x = torch.arange(-(Kw // 2), Kw // 2 + 1, dtype=torch.float32)
        if not Kw % 2:
            range_x = torch.cat((range_x[:Kw // 2], range_x[Kw // 2 + 1:]))

        range_y = torch.arange(-(Kh // 2), Kh // 2 + 1, dtype=torch.float32)
        if not Kh % 2:
            range_y = torch.cat((range_y[:Kh // 2], range_y[Kh // 2 + 1:]))

        kerX = torch.tan(range_x * delta_lon)
        kerY = torch.tan(range_y * delta_lat) / torch.cos(range_y * delta_lon)

        return torch.meshgrid(kerX, kerY, indexing="ij")  # (Kh, Kw)
