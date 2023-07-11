// See https://aka.ms/new-console-template for more information
using Microsoft.ML;
using MLPlayground.DataStructures;

List<string> filenames = new List<string> { "IMDB1K.tsv", "IMDB10K.tsv", "IMDB50K.tsv" };

Console.WriteLine("Select the dataset you want to use:");
for (int i = 0; i < filenames.Count; i++)
{
    Console.WriteLine($"{i + 1}. {filenames[i]}");
}

var input = Console.ReadLine();
int selectedIndex;
if (!int.TryParse(input, out selectedIndex) || selectedIndex < 1 || selectedIndex > filenames.Count)
{
    selectedIndex = 1;  // Default to the first file
}

string dataPath = filenames[selectedIndex - 1];

var mlContext = new MLContext();
IDataView dataView = mlContext.Data.LoadFromTextFile<SentimentIssue>(dataPath, hasHeader: true);

var tt = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
var trainingData = tt.TrainSet;
var testData = tt.TestSet;

var pipeline = mlContext.Transforms.Text.FeaturizeText("Features", nameof(SentimentIssue.Text))
    .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));

Console.WriteLine($"Training...");
var model = pipeline.Fit(trainingData);

var predictions = model.Transform(testData);
var metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");
Console.WriteLine($"Model accuracy on unseen test data: {Math.Round(metrics.Accuracy, 6)*100}%");


var predictionEngine = mlContext.Model.CreatePredictionEngine<SentimentIssue, SentimentPrediction>(model);

while (true)
{
    Console.WriteLine("Enter a text for sentiment analysis (or 'exit' to quit):");
    var inputText = Console.ReadLine();

    if (inputText?.ToLower() == "exit")
    {
        break;
    }

    var prediction = predictionEngine.Predict(new SentimentIssue { Text = inputText });

    Console.WriteLine($"Text: {inputText}\nIs Toxic: {prediction.Prediction}\nProbability: {prediction.Score}");
}
