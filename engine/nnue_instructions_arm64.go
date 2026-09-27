//go:build arm64 && cgo
// +build arm64,cgo

package engine

/*
#include <stdint.h>

void zahak_update_hidden(
    const float *previous_outputs,
    const int16_t *update_indices,
    const int8_t *update_coeffs,
    int update_size,
    const float *weights,
    float *outputs,
    int outputs_len);

float zahak_quick_feed(
    const float *hidden_outputs,
    int hidden_outputs_len,
    const float *weights,
    int weights_len);
*/
import "C"
import "unsafe"

func (n *NetworkState) UpdateHidden(updates *Updates) {
	n.CurrentHidden += 1

	C.zahak_update_hidden(
		(*C.float)(unsafe.Pointer(&n.HiddenOutputs[n.CurrentHidden-1][0])),
		(*C.int16_t)(unsafe.Pointer(&updates.Indices[0])),
		(*C.int8_t)(unsafe.Pointer(&updates.Coeffs[0])),
		C.int(updates.Size),
		(*C.float)(unsafe.Pointer(&n.HiddenWeights[0])),
		(*C.float)(unsafe.Pointer(&n.HiddenOutputs[n.CurrentHidden][0])),
		C.int(NetHiddenSize),
	)
}

func (n *NetworkState) QuickFeed() float32 {
	result := C.zahak_quick_feed(
		(*C.float)(unsafe.Pointer(&n.HiddenOutputs[n.CurrentHidden][0])),
		C.int(NetHiddenSize),
		(*C.float)(unsafe.Pointer(&n.OutputWeights[0])),
		C.int(NetHiddenSize),
	)
	return float32(result) + n.OutputBias
}
