package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"time"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

// Constants
const (
	ModelPath        = "../saved_models/lstm_BTCUSDT_1h/model" // Path to the saved TensorFlow model
	DataPath         = "../data/BTCUSDT_1h.csv"               // Path to the data CSV file
	SequenceLength   = 60                                      // Sequence length used for LSTM input
	BatchSize        = 1                                       // Batch size for inference
	PredictionLength = 1                                       // Number of steps to predict
)

// OHLCVData represents OHLCV (Open, High, Low, Close, Volume) data
type OHLCVData struct {
	Timestamp time.Time
	Open      float64
	High      float64
	Low       float64
	Close     float64
	Volume    float64
}

// PredictionResult represents the prediction result from the model
type PredictionResult struct {
	Timestamp  time.Time
	ActualPrice float64
	Prediction float64
	Signal     string
}

func main() {
	// Load the TensorFlow model
	model, err := loadModel(ModelPath)
	if err != nil {
		log.Fatalf("Error loading model: %v", err)
	}

	// Load historical data
	data, err := loadHistoricalData(DataPath)
	if err != nil {
		log.Fatalf("Error loading data: %v", err)
	}

	fmt.Printf("Loaded %d data points\n", len(data))

	// Make predictions
	predictions, err := makePredictions(model, data)
	if err != nil {
		log.Fatalf("Error making predictions: %v", err)
	}

	// Display predictions
	for i, pred := range predictions {
		if i < 10 { // Just show first 10 predictions
			fmt.Printf("Date: %s, Actual: %.2f, Predicted: %.2f, Signal: %s\n",
				pred.Timestamp.Format("2006-01-02 15:04:05"),
				pred.ActualPrice,
				pred.Prediction,
				pred.Signal)
		}
	}

	// Calculate accuracy
	accuracy := calculateAccuracy(predictions)
	fmt.Printf("\nDirectional Accuracy: %.2f%%\n", accuracy*100)
}

// loadModel loads the saved TensorFlow model
func loadModel(modelPath string) (*tf.SavedModel, error) {
	model, err := tf.LoadSavedModel(modelPath, []string{"serve"}, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to load model: %v", err)
	}

	return model, nil
}

// loadHistoricalData loads historical OHLCV data from a CSV file
func loadHistoricalData(csvPath string) ([]OHLCVData, error) {
	file, err := os.Open(csvPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %v", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)

	// Read header
	header, err := reader.Read()
	if err != nil {
		return nil, fmt.Errorf("failed to read header: %v", err)
	}

	// Map column indices
	colMap := make(map[string]int)
	for i, col := range header {
		colMap[col] = i
	}

	// Check required columns
	requiredCols := []string{"timestamp", "open", "high", "low", "close", "volume"}
	for _, col := range requiredCols {
		if _, ok := colMap[col]; !ok {
			// Try with capitalized column names
			if _, ok := colMap[capitalizeFirst(col)]; !ok {
				return nil, fmt.Errorf("required column not found: %s", col)
			} else {
				colMap[col] = colMap[capitalizeFirst(col)]
			}
		}
	}

	var data []OHLCVData

	// Read data rows
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("failed to read record: %v", err)
		}

		// Parse timestamp
		var timestamp time.Time
		timestampStr := record[colMap["timestamp"]]

		// Try different timestamp formats
		formats := []string{
			"2006-01-02 15:04:05",
			"2006-01-02T15:04:05Z",
			"2006-01-02",
		}

		parsed := false
		for _, format := range formats {
			if t, err := time.Parse(format, timestampStr); err == nil {
				timestamp = t
				parsed = true
				break
			}
		}

		if !parsed {
			// Try parsing as Unix timestamp
			if msec, err := strconv.ParseInt(timestampStr, 10, 64); err == nil {
				if len(timestampStr) > 10 { // Milliseconds
					timestamp = time.Unix(msec/1000, (msec%1000)*1000000)
				} else { // Seconds
					timestamp = time.Unix(msec, 0)
				}
			} else {
				return nil, fmt.Errorf("failed to parse timestamp: %s", timestampStr)
			}
		}

		// Parse OHLCV values
		open, _ := strconv.ParseFloat(record[colMap["open"]], 64)
		high, _ := strconv.ParseFloat(record[colMap["high"]], 64)
		low, _ := strconv.ParseFloat(record[colMap["low"]], 64)
		close, _ := strconv.ParseFloat(record[colMap["close"]], 64)
		volume, _ := strconv.ParseFloat(record[colMap["volume"]], 64)

		data = append(data, OHLCVData{
			Timestamp: timestamp,
			Open:      open,
			High:      high,
			Low:       low,
			Close:     close,
			Volume:    volume,
		})
	}

	return data, nil
}

// makePredictions generates predictions using the loaded model
func makePredictions(model *tf.SavedModel, data []OHLCVData) ([]PredictionResult, error) {
	if len(data) < SequenceLength+1 {
		return nil, fmt.Errorf("not enough data points for prediction")
	}

	var predictions []PredictionResult

	// Find model input and output operations
	inputOp, found := findOperation(model, "serving_default_input")
	if !found {
		// Try alternate input operation names
		alternateInputNames := []string{
			"serving_default_lstm_input",
			"serving_default_input_1",
			"serving_default_inputs",
		}

		for _, name := range alternateInputNames {
			if inputOp, found = findOperation(model, name); found {
				break
			}
		}

		if !found {
			return nil, fmt.Errorf("could not find input operation in model")
		}
	}

	outputOp, found := findOperation(model, "StatefulPartitionedCall")
	if !found {
		// Try alternate output operation names
		alternateOutputNames := []string{
			"PartitionedCall",
			"StatefulPartitionedCall:0",
			"Identity",
			"output",
		}

		for _, name := range alternateOutputNames {
			if outputOp, found = findOperation(model, name); found {
				break
			}
		}

		if !found {
			return nil, fmt.Errorf("could not find output operation in model")
		}
	}

	// Normalize data (min-max scaling to [0,1])
	normalizedData, err := normalizeData(data)
	if err != nil {
		return nil, fmt.Errorf("failed to normalize data: %v", err)
	}

	// Generate predictions for each point after sequence_length
	for i := SequenceLength; i < len(data); i++ {
		// Prepare input sequence
		sequence := normalizedData[i-SequenceLength:i]
		inputTensor, err := prepareInputTensor(sequence)
		if err != nil {
			return nil, fmt.Errorf("failed to prepare input tensor: %v", err)
		}

		// Run inference
		feeds := map[tf.Output]*tf.Tensor{
			inputOp: inputTensor,
		}

		fetches, err := model.Session.Run(
			feeds,
			[]tf.Output{outputOp},
			nil,
		)

		if err != nil {
			return nil, fmt.Errorf("failed to run inference: %v", err)
		}

		// Parse prediction
		if len(fetches) == 0 {
			return nil, fmt.Errorf("no output from model")
		}

		// Extract prediction value
		predictionValue, err := getPredictionValue(fetches[0])
		if err != nil {
			return nil, fmt.Errorf("failed to extract prediction: %v", err)
		}

		// Denormalize prediction
		denormalizedPrediction := denormalizePrediction(predictionValue, data)

		// Determine trading signal
		var signal string
		if denormalizedPrediction > data[i].Close {
			signal = "BUY"
		} else if denormalizedPrediction < data[i].Close {
			signal = "SELL"
		} else {
			signal = "HOLD"
		}

		// Add to predictions
		predictions = append(predictions, PredictionResult{
			Timestamp:   data[i].Timestamp,
			ActualPrice: data[i].Close,
			Prediction:  denormalizedPrediction,
			Signal:      signal,
		})
	}

	return predictions, nil
}

// findOperation finds a TensorFlow operation by name
func findOperation(model *tf.SavedModel, name string) (tf.Output, bool) {
	for _, op := range model.Graph.Operations() {
		if op.Name() == name {
			return op.Output(0), true
		}
	}
	return tf.Output{}, false
}

// normalizeData performs min-max normalization on the data
func normalizeData(data []OHLCVData) ([][]float64, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("empty data for normalization")
	}

	// Find min/max values for each feature
	minOpen, maxOpen := data[0].Open, data[0].Open
	minHigh, maxHigh := data[0].High, data[0].High
	minLow, maxLow := data[0].Low, data[0].Low
	minClose, maxClose := data[0].Close, data[0].Close
	minVolume, maxVolume := data[0].Volume, data[0].Volume

	for _, d := range data {
		// Update min values
		if d.Open < minOpen {
			minOpen = d.Open
		}
		if d.High < minHigh {
			minHigh = d.High
		}
		if d.Low < minLow {
			minLow = d.Low
		}
		if d.Close < minClose {
			minClose = d.Close
		}
		if d.Volume < minVolume {
			minVolume = d.Volume
		}

		// Update max values
		if d.Open > maxOpen {
			maxOpen = d.Open
		}
		if d.High > maxHigh {
			maxHigh = d.High
		}
		if d.Low > maxLow {
			maxLow = d.Low
		}
		if d.Close > maxClose {
			maxClose = d.Close
		}
		if d.Volume > maxVolume {
			maxVolume = d.Volume
		}
	}

	// Normalize data
	normalizedData := make([][]float64, len(data))

	for i, d := range data {
		normalizedData[i] = []float64{
			normalizeValue(d.Open, minOpen, maxOpen),
			normalizeValue(d.High, minHigh, maxHigh),
			normalizeValue(d.Low, minLow, maxLow),
			normalizeValue(d.Close, minClose, maxClose),
			normalizeValue(d.Volume, minVolume, maxVolume),
		}
	}

	return normalizedData, nil
}

// normalizeValue performs min-max normalization on a single value
func normalizeValue(value, min, max float64) float64 {
	if max == min {
		return 0.5 // Avoid division by zero
	}
	return (value - min) / (max - min)
}

// prepareInputTensor prepares the input tensor for model inference
func prepareInputTensor(sequence [][]float64) (*tf.Tensor, error) {
	// Convert to [1, sequence_length, num_features] tensor
	numFeatures := len(sequence[0])
	inputShape := []int64{1, int64(len(sequence)), int64(numFeatures)}

	// Flatten the sequence for tensor creation
	flattenedSequence := make([]float32, 0, len(sequence)*numFeatures)
	for _, s := range sequence {
		for _, v := range s {
			flattenedSequence = append(flattenedSequence, float32(v))
		}
	}

	tensor, err := tf.NewTensor(flattenedSequence)
	if err != nil {
		return nil, fmt.Errorf("failed to create tensor: %v", err)
	}

	// Reshape tensor to expected input shape
	scope := op.NewScope()
	input := op.Placeholder(scope, tf.Float, op.PlaceholderShape(tf.MakeShape(int64(len(flattenedSequence)))))
	reshape := op.Reshape(scope, input, op.Const(scope.SubScope("reshape"), inputShape))

	graph, err := scope.Finalize()
	if err != nil {
		return nil, fmt.Errorf("failed to finalize graph: %v", err)
	}

	sess, err := tf.NewSession(graph, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create session: %v", err)
	}
	defer sess.Close()

	feeds := map[tf.Output]*tf.Tensor{
		input: tensor,
	}

	fetches, err := sess.Run(feeds, []tf.Output{reshape}, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to run reshape: %v", err)
	}

	return fetches[0], nil
}

// getPredictionValue extracts the prediction value from the output tensor
func getPredictionValue(tensor *tf.Tensor) (float64, error) {
	// Extract predicted value based on tensor shape and type
	switch tensor.Shape().Size() {
	case 1:
		// Single scalar value
		value, ok := tensor.Value().(float32)
		if !ok {
			return 0, fmt.Errorf("unexpected tensor type, expected float32, got %T", tensor.Value())
		}
		return float64(value), nil

	case 2:
		// [batch_size, 1] tensor
		values, ok := tensor.Value().([][]float32)
		if !ok {
			return 0, fmt.Errorf("unexpected tensor type, expected [][]float32, got %T", tensor.Value())
		}
		if len(values) == 0 || len(values[0]) == 0 {
			return 0, fmt.Errorf("empty prediction tensor")
		}
		return float64(values[0][0]), nil

	default:
		// Try to extract value from tensor with unknown shape
		value := tensor.Value()

		switch v := value.(type) {
		case float32:
			return float64(v), nil
		case float64:
			return v, nil
		case []float32:
			if len(v) > 0 {
				return float64(v[0]), nil
			}
		case []float64:
			if len(v) > 0 {
				return v[0], nil
			}
		case [][]float32:
			if len(v) > 0 && len(v[0]) > 0 {
				return float64(v[0][0]), nil
			}
		case [][]float64:
			if len(v) > 0 && len(v[0]) > 0 {
				return v[0][0], nil
			}
		}

		return 0, fmt.Errorf("unable to extract prediction value from tensor")
	}
}

// denormalizePrediction converts the normalized prediction back to original scale
func denormalizePrediction(normalizedValue float64, data []OHLCVData) float64 {
	// Find min/max values for close price
	minClose, maxClose := data[0].Close, data[0].Close

	for _, d := range data {
		if d.Close < minClose {
			minClose = d.Close
		}
		if d.Close > maxClose {
			maxClose = d.Close
		}
	}

	// Denormalize
	return normalizedValue*(maxClose-minClose) + minClose
}

// calculateAccuracy calculates the directional accuracy of predictions
func calculateAccuracy(predictions []PredictionResult) float64 {
	if len(predictions) <= 1 {
		return 0
	}

	correct := 0
	total := len(predictions) - 1

	for i := 1; i < len(predictions); i++ {
		actualDirection := predictions[i].ActualPrice > predictions[i-1].ActualPrice
		predictedDirection := predictions[i].Prediction > predictions[i-1].ActualPrice

		if actualDirection == predictedDirection {
			correct++
		}
	}

	return float64(correct) / float64(total)
}

// capitalizeFirst capitalizes the first letter of a string
func capitalizeFirst(s string) string {
	if s == "" {
		return s
	}
	r := []rune(s)
	r[0] = []rune(fmt.Sprintf("%c", r[0]))[0] - 32
	return string(r)
}