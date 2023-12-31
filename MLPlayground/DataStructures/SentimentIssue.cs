﻿using Microsoft.ML.Data;

namespace MLPlayground.DataStructures
{
    public class SentimentIssue
    {
        [LoadColumn(0)]
        public bool Label { get; set; }

        [LoadColumn(1)]
        public string? Text { get; set; }
    }
}