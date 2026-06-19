package rbtz

import (
	"encoding/binary"
	"errors"
)

// See https://pkg.go.dev/github.com/SaveTheRbtz/mph

// Copyright (c) 2016 Caleb Spare
// Copyright (c) 2022 Alexey Ivanov
//
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

// code adopted by Ogurtsov Andrei

func (t *Table) ByteSize() int {
	return 8 + len(t.level0)*4 + 8 + len(t.level1)*4
}

func (t *Table) Serialize() ([]byte, error) {
	size := t.ByteSize()
	buf := make([]byte, size)
	offset := 0

	binary.LittleEndian.PutUint64(buf[offset:], uint64(len(t.level0)))
	offset += 8

	for _, v := range t.level0 {
		binary.LittleEndian.PutUint32(buf[offset:], v)
		offset += 4
	}

	binary.LittleEndian.PutUint64(buf[offset:], uint64(len(t.level1)))
	offset += 8

	for _, v := range t.level1 {
		binary.LittleEndian.PutUint32(buf[offset:], v)
		offset += 4
	}

	return buf, nil
}

func Deserialize(data []byte, t *Table) error {
	if len(data) < 8 {
		return errors.New("mph: data too short for level0 length")
	}

	offset := 0
	l0Len := int(binary.LittleEndian.Uint64(data[offset:]))
	offset += 8

	if len(data[offset:]) < l0Len*4 {
		return errors.New("mph: data too short for level0 content")
	}

	t.level0 = make([]uint32, l0Len)
	for i := 0; i < l0Len; i++ {
		t.level0[i] = binary.LittleEndian.Uint32(data[offset:])
		offset += 4
	}
	t.level0Mask = l0Len - 1

	if len(data[offset:]) < 8 {
		return errors.New("mph: data too short for level1 length")
	}

	l1Len := int(binary.LittleEndian.Uint64(data[offset:]))
	offset += 8

	if len(data[offset:]) < l1Len*4 {
		return errors.New("mph: data too short for level1 content")
	}

	t.level1 = make([]uint32, l1Len)
	for i := 0; i < l1Len; i++ {
		t.level1[i] = binary.LittleEndian.Uint32(data[offset:])
		offset += 4
	}
	t.level1Mask = l1Len - 1

	return nil
}
