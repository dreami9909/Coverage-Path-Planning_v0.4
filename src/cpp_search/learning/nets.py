"""Su & Qian 2023 의 신경망 구조를 numpy 로 만든다.

왜 직접 만드는가
----------------
이 저장소는 의존성이 numpy 뿐이다. 그런데 원문의 기본 MAPPO 는 선형
softmax 가 아니다 — 논문 §5.2 (Figure 2) 가 구조를 글자로 적어놓았다.

    Conv2D -> ReLU -> Flatten -> FC -> ReLU -> FC -> ReLU -> FC -> Softmax

(재귀 변형은 마지막 FC 앞에 GRU 를 넣는다. 논문 §6.3 은 **기본 구조가 더
낫다**고 보고하므로 우리가 맞춰야 할 것은 기본 구조다.)

선형 정책으로는 "표적 확률이 높으면서 **동시에** 오래 안 본 곳" 같은 채널
사이의 곱 항을 표현할 수 없다. 세 채널을 각각 가중합할 뿐이다. 그래서
"논문 성능을 재현했다" 고 말할 수 없었다. 이 모듈이 그 제약을 없앤다.

논문이 적지 않은 것
-------------------
커널 크기, 채널 수, FC 폭은 논문에 없다. 그러므로 **선언하고 기록한다** —
config 의 ``cnn`` 블록이 정하고 결과 JSON 의 ``policy_capacity`` 에 남는다.
"논문과 같은 구조" 라고는 말할 수 있어도 "논문과 같은 용량" 이라고는 말할
수 없고, 그 구분을 결과가 스스로 밝히게 한다.

역전파는 손으로 적었으므로 **유한차분으로 검증한다** —
``tests/test_chapter5_mappo.py`` 가 모든 파라미터에 대해 상대오차를 잰다.
검증 없는 수동 역전파는 틀려도 학습이 '돌아가기' 때문에 곡선만 봐서는
잡히지 않는다.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _im2col(
    batch: np.ndarray, kernel_h: int, kernel_w: int, stride: int
) -> tuple[np.ndarray, int, int]:
    """``(N, C, H, W)`` -> ``(N, C*kh*kw, OH*OW)``.

    합성곱을 행렬곱 하나로 바꾼다. 파이썬 반복문으로 픽셀을 돌면 이 규모
    (에피소드당 200 스텝 x 500 에피소드)에서 감당이 안 된다.
    """

    count, channels, height, width = batch.shape
    out_h = (height - kernel_h) // stride + 1
    out_w = (width - kernel_w) // stride + 1
    if out_h <= 0 or out_w <= 0:
        raise ValueError("kernel is larger than the input")
    # stride_tricks 로 겹치는 창을 복사 없이 본 뒤 한 번만 reshape 한다.
    strides = batch.strides
    windows = np.lib.stride_tricks.as_strided(
        batch,
        shape=(count, channels, out_h, out_w, kernel_h, kernel_w),
        strides=(
            strides[0],
            strides[1],
            strides[2] * stride,
            strides[3] * stride,
            strides[2],
            strides[3],
        ),
        writeable=False,
    )
    columns = windows.transpose(0, 1, 4, 5, 2, 3).reshape(
        count, channels * kernel_h * kernel_w, out_h * out_w
    )
    return columns, out_h, out_w


class Conv2D:
    """``valid`` 패딩, 정사각 커널, 임의 stride 의 2차원 합성곱.

    이 신경망에서 Conv2D 는 **항상 첫 층**이므로 입력에 대한 그래디언트는
    쓸 데가 없다. 그래서 ``backward`` 는 파라미터 그래디언트만 돌려주고
    col2im 을 하지 않는다 — 안 쓰는 것을 계산하지 않는 편이 정직하다.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        rng: np.random.Generator,
        name: str,
    ) -> None:
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.name = name
        fan_in = self.in_channels * self.kernel_size * self.kernel_size
        # He 초기화. 뒤에 ReLU 가 붙으므로 절반이 죽는 것을 감안해 2/fan_in.
        self.weight = rng.normal(
            0.0,
            np.sqrt(2.0 / fan_in),
            (self.out_channels, fan_in),
        )
        self.bias = np.zeros(self.out_channels, dtype=float)

    def output_shape(self, input_shape: Sequence[int]) -> tuple[int, int, int]:
        _, height, width = input_shape
        out_h = (height - self.kernel_size) // self.stride + 1
        out_w = (width - self.kernel_size) // self.stride + 1
        return self.out_channels, out_h, out_w

    @property
    def parameters(self) -> dict[str, np.ndarray]:
        return {f"{self.name}.weight": self.weight, f"{self.name}.bias": self.bias}

    def forward(self, batch: np.ndarray) -> tuple[np.ndarray, dict]:
        columns, out_h, out_w = _im2col(
            batch, self.kernel_size, self.kernel_size, self.stride
        )
        # (N, F, OH*OW) = (F, C*kh*kw) @ (N, C*kh*kw, OH*OW)
        #
        # ``matmul`` 은 앞의 배치 축을 브로드캐스트하면서 BLAS 를 쓴다.
        # 같은 식을 einsum 으로 쓰면 일반 경로로 떨어지는데, 갱신 프로파일에서
        # einsum 이 전체의 63% 였다.
        activation = np.matmul(self.weight, columns)
        activation += self.bias[None, :, None]
        output = activation.reshape(batch.shape[0], self.out_channels, out_h, out_w)
        return output, {"columns": columns, "out_h": out_h, "out_w": out_w}

    def backward(
        self, gradient: np.ndarray, cache: dict
    ) -> tuple[None, dict[str, np.ndarray]]:
        count = gradient.shape[0]
        flat = gradient.reshape(count, self.out_channels, -1)
        # sum over 배치(n)와 출력위치(p): (F, D). tensordot 은 내부에서
        # 한 번 reshape 한 뒤 BLAS gemm 을 부른다.
        weight_gradient = np.tensordot(
            flat, cache["columns"], axes=([0, 2], [0, 2])
        )
        bias_gradient = flat.sum(axis=(0, 2))
        return None, {
            f"{self.name}.weight": weight_gradient,
            f"{self.name}.bias": bias_gradient,
        }

    def set(self, name: str, value: np.ndarray) -> None:
        if name == f"{self.name}.weight":
            self.weight = value
        elif name == f"{self.name}.bias":
            self.bias = value
        else:  # pragma: no cover - 이름 사고를 조용히 넘기지 않는다
            raise KeyError(name)


class Dense:
    """``(..., in) -> (..., out)`` 완전연결층."""

    def __init__(
        self,
        in_size: int,
        out_size: int,
        *,
        rng: np.random.Generator,
        name: str,
        relu_next: bool = True,
    ) -> None:
        self.in_size = int(in_size)
        self.out_size = int(out_size)
        self.name = name
        scale = np.sqrt((2.0 if relu_next else 1.0) / max(self.in_size, 1))
        self.weight = rng.normal(0.0, scale, (self.in_size, self.out_size))
        self.bias = np.zeros(self.out_size, dtype=float)

    @property
    def parameters(self) -> dict[str, np.ndarray]:
        return {f"{self.name}.weight": self.weight, f"{self.name}.bias": self.bias}

    def forward(self, batch: np.ndarray) -> tuple[np.ndarray, dict]:
        return batch @ self.weight + self.bias, {"input": batch}

    def backward(
        self, gradient: np.ndarray, cache: dict
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        batch = cache["input"]
        # 선행 축이 몇 개든 (샘플, 기체) 두 축일 수도 있다 — 마지막 축만
        # 특징 차원으로 보고 나머지는 전부 합친다.
        flat_input = batch.reshape(-1, self.in_size)
        flat_gradient = gradient.reshape(-1, self.out_size)
        return gradient @ self.weight.T, {
            f"{self.name}.weight": flat_input.T @ flat_gradient,
            f"{self.name}.bias": flat_gradient.sum(axis=0),
        }

    def set(self, name: str, value: np.ndarray) -> None:
        if name == f"{self.name}.weight":
            self.weight = value
        elif name == f"{self.name}.bias":
            self.bias = value
        else:  # pragma: no cover
            raise KeyError(name)


class ReLU:
    name = "relu"
    parameters: dict[str, np.ndarray] = {}

    def forward(self, batch: np.ndarray) -> tuple[np.ndarray, dict]:
        mask = batch > 0.0
        return batch * mask, {"mask": mask}

    def backward(self, gradient: np.ndarray, cache: dict) -> tuple[np.ndarray, dict]:
        return gradient * cache["mask"], {}

    def set(self, name: str, value: np.ndarray) -> None:  # pragma: no cover
        raise KeyError(name)


class Flatten:
    """앞쪽 ``leading`` 개 축은 남기고 나머지를 하나로 편다."""

    name = "flatten"
    parameters: dict[str, np.ndarray] = {}

    def __init__(self, leading: int = 1) -> None:
        self.leading = int(leading)

    def forward(self, batch: np.ndarray) -> tuple[np.ndarray, dict]:
        shape = batch.shape
        flat = batch.reshape(*shape[: self.leading], -1)
        return flat, {"shape": shape}

    def backward(self, gradient: np.ndarray, cache: dict) -> tuple[np.ndarray, dict]:
        return gradient.reshape(cache["shape"]), {}

    def set(self, name: str, value: np.ndarray) -> None:  # pragma: no cover
        raise KeyError(name)


class ConcatTail:
    """격자를 지난 특징 뒤에 벡터 꼬리를 붙인다.

    비평자의 전역상태는 ``[격자 채널..., 기체 위치...]`` 다. 위치는 격자가
    아니므로 합성곱에 넣을 수 없고, 그렇다고 버릴 수도 없다 (중앙화 비평자가
    보는 정보의 일부다). 합성곱을 지난 뒤 이어 붙인다.
    """

    name = "concat_tail"
    parameters: dict[str, np.ndarray] = {}

    def __init__(self, tail_size: int) -> None:
        self.tail_size = int(tail_size)
        self._tail: np.ndarray | None = None

    def set_tail(self, tail: np.ndarray) -> None:
        self._tail = tail

    def forward(self, batch: np.ndarray) -> tuple[np.ndarray, dict]:
        if self.tail_size == 0:
            return batch, {"width": batch.shape[-1]}
        if self._tail is None:
            raise RuntimeError("tail was not supplied before the forward pass")
        return np.concatenate([batch, self._tail], axis=-1), {"width": batch.shape[-1]}

    def backward(self, gradient: np.ndarray, cache: dict) -> tuple[np.ndarray, dict]:
        return gradient[..., : cache["width"]], {}

    def set(self, name: str, value: np.ndarray) -> None:  # pragma: no cover
        raise KeyError(name)


class ConvNet:
    """Conv2D -> ReLU -> Flatten -> (꼬리 결합) -> [FC -> ReLU] x n -> FC.

    출력에 활성함수를 붙이지 않는다. 행위자는 밖에서 softmax 를, 비평자는
    그대로 값으로 쓴다 — 원문 Figure 2 의 (a)/(b) 차이가 그것뿐이다.
    """

    def __init__(
        self,
        input_shape: Sequence[int],
        output_size: int,
        *,
        conv_channels: int,
        kernel_size: int,
        stride: int,
        fully_connected: Sequence[int],
        tail_size: int = 0,
        rng: np.random.Generator,
        prefix: str,
    ) -> None:
        self.input_shape = tuple(int(v) for v in input_shape)
        self.tail_size = int(tail_size)
        conv = Conv2D(
            self.input_shape[0],
            conv_channels,
            kernel_size,
            stride=stride,
            rng=rng,
            name=f"{prefix}.conv",
        )
        channels, out_h, out_w = conv.output_shape(self.input_shape)
        self.concat = ConcatTail(self.tail_size)
        width = channels * out_h * out_w + self.tail_size
        layers: list = [conv, ReLU(), Flatten(leading=1), self.concat]
        for index, size in enumerate(fully_connected):
            layers.append(
                Dense(width, size, rng=rng, name=f"{prefix}.fc{index}", relu_next=True)
            )
            layers.append(ReLU())
            width = size
        layers.append(
            Dense(
                width,
                output_size,
                rng=rng,
                name=f"{prefix}.head",
                relu_next=False,
            )
        )
        self.layers = layers
        self.output_size = int(output_size)
        self.conv_output_shape = (channels, out_h, out_w)

    @property
    def parameters(self) -> dict[str, np.ndarray]:
        merged: dict[str, np.ndarray] = {}
        for layer in self.layers:
            merged.update(layer.parameters)
        return merged

    def set(self, name: str, value: np.ndarray) -> None:
        for layer in self.layers:
            if name in layer.parameters:
                layer.set(name, value)
                return
        raise KeyError(name)

    def forward(self, grid: np.ndarray, tail: np.ndarray | None = None):
        """``grid`` 는 ``(N, C, H, W)``, ``tail`` 은 ``(N, tail_size)``."""

        if self.tail_size:
            if tail is None:
                raise ValueError("this network expects a vector tail")
            self.concat.set_tail(tail)
        activation = grid
        caches = []
        for layer in self.layers:
            activation, cache = layer.forward(activation)
            caches.append(cache)
        return activation, caches

    def backward(self, gradient: np.ndarray, caches) -> dict[str, np.ndarray]:
        gradients: dict[str, np.ndarray] = {}
        for layer, cache in zip(reversed(self.layers), reversed(caches)):
            gradient, parameter_gradients = layer.backward(gradient, cache)
            gradients.update(parameter_gradients)
            if gradient is None:
                break
        return gradients
