//
//  model.swift
//  swft_model
//
//  Created by Julien VIGUIER on 04/05/2018.
//  Copyright © 2018 Julien VIGUIER. All rights reserved.
//

import AppKit
import Foundation
import TensorFlow

struct Pixel {
    var r: Float
    var g: Float
    var b: Float
    var a: Float
    var row: Int
    var col: Int
    
    init(r: UInt8, g: UInt8, b: UInt8, a: UInt8, row: Int, col: Int) {
        self.r = Float(r)
        self.g = Float(g)
        self.b = Float(b)
        self.a = Float(a)
        self.row = row
        self.col = col
    }
}

func pixelToFloat(_pixelTab: [[Pixel]]) -> [Float] {
    var tab: [Float] = []
    for (_, row) in _pixelTab.enumerated() {
        for (_, cell) in row.enumerated() {
            tab.append(Float(cell.r/255.0))
            tab.append(Float(cell.g/255.0))
            tab.append(Float(cell.b/255.0))
        }
    }
    return tab
}

func pixelData(bmp: NSBitmapImageRep) -> [Pixel] {
    var pixels: [Pixel] = []
    var data: UnsafeMutablePointer<UInt8> = bmp.bitmapData!
    var r, g, b, a: UInt8
    
    for row in 0..<bmp.pixelsHigh {
        for col in 0..<bmp.pixelsWide {
            r = data.pointee
            data = data.advanced(by: 1)
            g = data.pointee
            data = data.advanced(by: 1)
            b = data.pointee
            data = data.advanced(by: 1)
            a = data.pointee
            pixels.append(Pixel(r: r, g: g, b: b, a: a, row: row, col: col))
        }
    }
    return pixels
}

@inline(never)
public func getLoss(pred: Tensor<Float>, expect: Tensor<Float>) -> Float {
    return (pred - expect).squared().mean(alongAxes: 0, 1).scalarized()
}

@inline(never)
public func getSoftmax(input: Tensor<Float>) -> Tensor<Float> {
    return softmax(input).toDevice()
}

@inline(never)
public func getArgmax(input: Tensor<Float>, axe: Int32) -> Tensor<Int32> {
    return (input.argmax(squeezingAxis: axe)).toDevice()
}

@inline(never)
public func getSigmoid(input: Tensor<Float>) -> Tensor<Float> {
    return sigmoid(input).toDevice()
}

@inline(never)
public func conv2d(input: Tensor<Float>, _filter: Tensor<Float>) -> Tensor<Float> {
    return input.convolved2D(withFilter: _filter, strides: (1, 1, 1, 1), padding: Padding.same).toDevice()
}

@inline(never)
public func matMul(a: Tensor<Float>, b: Tensor<Float>) -> Tensor<Float> {
    return (a ⊗ b).toDevice()
}

@inline(never)
public func matAdd(a: Tensor<Float>, b: Tensor<Float>) -> Tensor<Float> {
    return (a + b).toDevice()
}

@inline(never)
public func maxPool2d(input: Tensor<Float>, _kernel: Int32, _stride: Int32) -> Tensor<Float> {
    return input.maxPooled(kernelSize: (1, _kernel, _kernel, 1), strides: (1, _stride, _stride, 1), padding: Padding.same).toDevice()
}

@inline(never)
public func reluActivation(input: Tensor<Float>) -> Tensor<Float> {
    return relu(input).toDevice()
}

@inline(never)
public func matReshapeFP(_in: Tensor<Float>) -> Tensor<Float> {
    return _in.reshaped(to: [-1, Int32(4*4*128)]).toDevice()
}

@inline(never)
public func matReshapeBP(_in: Tensor<Float>) -> Tensor<Float> {
    return _in.reshaped(toShape: [1, 4, 4, 128]).toDevice()
}

@inline(never)
public func getTensorShape(_tensorShp: Tensor<Float>) -> TensorShape {
    return _tensorShp.shape
}

@inline(never)
public func createTensorOneHot(input: Tensor<Int32>, _depth: Int32) -> Tensor<Float> {
    let _tensorOH = Tensor<Float>(oneHotAtIndices: input, depth: _depth)
    return _tensorOH.toDevice()
}

@inline(never)
public func createTensorFloat2D(_row: Int32, _column: Int32, _scalars: [Float]) -> Tensor<Float> {
    let _tensorF = Tensor(shape: [_row, _column], scalars: _scalars)
    return _tensorF.toDevice()
}

@inline(never)
public func createTensorFloat3D(_row: Int32, _column: Int32, _depth: Int32, _scalars: [Float]) -> Tensor<Float> {
    let _tensorS = Tensor(shape: [_row, _column, _depth], scalars: _scalars)
    return _tensorS.toDevice()
}

@inline(never)
public func createTensorFloat4D(w: Int32, x: Int32, y: Int32, z: Int32, _scalars: [Float]) -> Tensor<Float> {
    let _tensorF4D = Tensor(shape: [w, x, y, z], scalars: _scalars)
    return _tensorF4D.toDevice()
}

@inline(never)
public func createTensorInt1D(_vector : [Int32]) -> Tensor<Int32> {
    let _tensorI = Tensor(_vector)
    return _tensorI.toDevice()
}

@inline(never)
public func createTensorZero1D(x: Int32) -> Tensor<Float> {
    let _tensorZ = Tensor<Float>(zeros: [x])
    return _tensorZ.toDevice()
}

@inline(never)
public func createTensorRU4D(w: Int32, x: Int32, y: Int32, z: Int32) -> Tensor<Float> {
    let _tensorRU4D = Tensor<Float>(randomUniform: [w, x, y, z])
    return _tensorRU4D.toDevice()
}

@inline(never)
public func createTensorRU2D(x: Int32, y: Int32) -> Tensor<Float> {
    let _tensorRU2D = Tensor<Float>(randomUniform: [x, y])
    return _tensorRU2D.toDevice()
}

@inline(never)
public func convNet(_input: Tensor<Float>, _weights: [Tensor<Float>], _biases: [Tensor<Float>], lRate: Float, itCount: Int32) -> Float {
    /*
        input:                      [1, 28, 28, 3]
        conv1 after conv2d:         [1, 28, 28, 32]             filter applied:         [3, 3, 3, 32]       -> w1
        conv1 after maxpool:        [1, 14, 14, 32]
        conv2 after conv2d:         [1, 14, 14, 64]             filter applied:         [3, 3, 32, 64]      -> w2
        conv2 after maxpool:        [1, 7, 7, 64]
        conv3 after conv2d:         [1, 7, 7, 128]              filter applied:         [3, 3, 64, 128]     -> w3
        conv3 after maxpool:        [1, 4, 4, 128]
        fc1 after reshape:          [1, 2048]
        fc2 after ⊗:                [1, 128]                    filter applied:         [2048, 128]         -> w4
        fc3 after +:                [1, 128]                    filter applied:         [128]               -> b4
        fc4 after relu:             [1, 128]
        fc5 after ⊗:                [1, 2]                      filter applied:         [128, 2]            -> w5
        fc6 after +:                [1, 2]                      filter applied:         [2]                 -> b5
    */
    
    var lossTotal: Float = 0.0
    var i: Int32 = 0
    
    repeat {
//        Forward pass
        let conv1: Tensor<Float> = conv2d(input: _input, _filter: _weights[0])
        let conv1A = matAdd(a: conv1, b: _biases[0])
        let conv1R = reluActivation(input: conv1A)
        let maxP1 = maxPool2d(input: conv1, _kernel: 2, _stride: 2)
        let conv2: Tensor<Float> = conv2d(input: maxP1, _filter: _weights[1])
        let conv2A = matAdd(a: conv2, b: _biases[1])
        let conv2R = reluActivation(input: conv2A)
        let maxP2 = maxPool2d(input: conv2, _kernel: 2, _stride: 2)
        let conv3: Tensor<Float> = conv2d(input: maxP2, _filter: _weights[2])
        let conv3A = matAdd(a: conv3, b: _biases[2])
        let conv3R = reluActivation(input: conv3A)
        let maxP3 = maxPool2d(input: conv3, _kernel: 2, _stride: 2)
        let fc1: Tensor<Float> = matReshapeFP(_in: maxP3)
        let fc2: Tensor<Float> = matMul(a: fc1, b: _weights[3])
        let fc3: Tensor<Float> = matAdd(a: fc2, b: _biases[3])
        let fc4 = reluActivation(input: fc3)
        let fc5: Tensor<Float> = matMul(a: fc4, b: _weights[4])
        let fc6: Tensor<Float> = matAdd(a: fc5, b: _biases[4])
        let out: Tensor<Float> = getSigmoid(input: fc6)
        
//        Backward pass
        let dOut = (1 - out) * out
        let (dFc5, dB4) = #adjoint(Tensor.+)(
            fc5, _biases[4], originalValue: fc6, seed: dOut
        )
        let (dFc4, dW4) = #adjoint(matmul)(
            fc4, _weights[4], originalValue: fc5, seed: dFc5
        )
        let dFc3 = #adjoint(relu)(
            fc3, originalValue: fc4, seed: dFc4
        )
        let (dFc2, dB3) = #adjoint(Tensor.+)(
            fc2, _biases[3], originalValue: fc3, seed: dFc3
        )
        let (dFc1, dW3) = #adjoint(matmul)(
            fc1, _weights[3], originalValue: fc2, seed: dFc2
        )
        let dFc1Rshp = #adjoint(reshaped)(
            originalValue: fc1, seed: dFc1
        )
        let dMaxP3 = #adjoint(maxPooled)(
            conv3, originalValue: maxP3, seed: dFc1Rshp
        )
        let (dConv3R) = #adjoint(relu)(
            conv3A, originalValue: conv3R, seed: dMaxP3
        )
        let (dConv3A, dB2) = #adjoint(Tensor.+)(
            conv3, _biases[2], originalValue: conv3A, seed: dConv3R
        )
        let (dConv3, dW2) = #adjoint(convolved2D)(
            maxP2, _weights[2], originalValue: conv3, seed: dConv3A
        )
        let dMaxP2 = #adjoint(maxPooled)(
            conv2, originalValue: maxP2, seed: dConv3
        )
        let (dConv2R) = #adjoint(relu)(
            conv2A, originalValue: conv2R, seed: dMaxP2
        )
        let (dConv2A, dB1) = #adjoint(Tensor.+)(
            conv2, _biases[1], originalValue: conv2A, seed: dConv2R
        )
        let (dConv2, dW1) = #adjoint(convolved2D)(
            maxP1, _weights[1], originalValue: conv2, seed: dConv2A
        )
        let dMaxP1 = #adjoint(maxPooled)(
            conv1, originalValue: maxP1, seed: dConv2
        )
        let (dConv1R) = #adjoint(relu)(
            conv2A, originalValue: conv1R, seed: dMaxP1
        )
        let (dConv1A, dB0) = #adjoint(Tensor.+)(
            conv1, _biases[0], originalValue: conv1A, seed: dConv1R
        )
        let (_, dW0) = #adjoint(convolved2D)(
            input, _weights[0], originalValue: conv1, seed: dMaxP1
        )
        
        let loss: Float = getLoss(pred: out, expect: _input)
        print("loss per iteration: \(loss)")
        lossTotal += loss/i
        
//        Gradient descent
        _weights[0] -= lRate * dW0
        _weights[1] -= lRate * dW1
        _weights[2] -= lRate * dW2
        _weights[3] -= lRate * dW3
        _weights[4] -= lRate * dW4 
        
        _biases[0] -= lRate * dB0
        _biases[1] -= lRate * dB1
        _biases[2] -= lRate * dB2
        _biases[3] -= lRate * dB3
        _biases[4] -= lRate * dB4
        
        i += 1
    } while i < itCount
    
    return lossTotal
}

@inline(never)
public func loadImg(imgDir: String, labelDir: String) -> (Tensor<Float>, Tensor<Int32>){
    let labelURL = URL(fileURLWithPath: labelDir)
    var tabPixel: [[Pixel]] = [[]]
    var i: Int32 = 0
    
    do {
        let textFile = try String(contentsOf: labelURL, encoding: .utf8)
        var labelTab = textFile.split(separator: "\n")
    } catch {
        print(error)
    }
    
    repeat {
        let x: String =  "\(i).png"
        let imageURL: URL = URL(fileURLWithPath: imgDir).appendingPathComponent(x)
        let image: NSImage? = NSImage(contentsOf: imageURL)
        let imgData: Data = image!.tiffRepresentation!
        let imgBitMap = NSBitmapImageRep(data: imgData)
        let pixelValue: [Pixel] = pixelData(bmp: imgBitMap!)
        labelTab.append(x)
        tabPixel.append(pixelValue)
        i += 1
    } while i < 1
    
    let pixelFloat: [Float] = pixelToFloat(_pixelTab: tabPixel)
    let _tensorImg: Tensor<Float> = createTensorFloat4D(w: 1, x: 28, y: 28, z: 3, _scalars: pixelFloat)
    let _tensorLabel: Tensor<Int32> = createTensorInt1D(_vector: tabInt)
    
    return (_tensorImg.toDevice(), _tensorLabel.toDevice())
}

func main() {
    let _imgDir: String = "/Users/julienviguier/Documents/swft_model/imgDataset"
    let _labelDir: String = "/Users/julienviguier/Documents/swft_model/labels.txt"
    
    let inputNbr: Int32 = 10
    
    let (img, labels): (Tensor<Float>, Tensor<Int32>) = loadImg(imgDir: _imgDir, labelDir: _labelDir)
    
//    print("img: \n\n\(img)\n")
//    print("labels: \n\n\(labels)\n")

    let label = createTensorOneHot(input: labels, _depth: inputNbr)
    
//    print("\nlabels One Hot: \(label)")
    
    let trainIt: Int32 = 200
    let _learnRate: Float = 0.001
    let batchSize: Int32 = 100
    
    var weights: [Tensor<Float>] = []
    var biases: [Tensor<Float>] = []
    
    let weight1 = createTensorRU4D(w: 3, x: 3, y: 3, z: 32)
    let weight2 = createTensorRU4D(w: 3, x: 3, y: 32, z: 64)
    let weight3 = createTensorRU4D(w: 3, x: 3, y: 64, z: 128)
    let weight4 = createTensorRU2D(x: 2048, y: 128)
    let weight5 = createTensorRU2D(x: 128, y: inputNbr)
    
    let biase1 = createTensorZero1D(x: 32)
    let biase2 = createTensorZero1D(x: 64)
    let biase3 = createTensorZero1D(x: 128)
    let biase4 = createTensorZero1D(x: 128)
    let biase5 = createTensorZero1D(x: inputNbr)
    
    weights.append(contentsOf: [weight1, weight2, weight3, weight4, weight5])
    biases.append(contentsOf: [biase1, biase2, biase3, biase4, biase5])
    
    let loss: Float = convNet(_input: img, _weights: weights, _biases: biases, lRate: _learnRate, itCount: trainIt)
}

main()
