package org.deeplearning4j.examples;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;

import java.io.File;
import java.io.IOException;
import java.util.Collection;

/**
 * Created by Shakti on 6/1/2016.
 */
public class MyExample2 {
    public static void main(String args[]) throws IOException {
        System.out.println("Loading google model");
        File gModel = new File("/home/shakti_cloudcompute/downloads/GoogleNews-vectors-negative300.bin");
        WordVectors wordVectors = WordVectorSerializer.loadGoogleModel(gModel, true);
        System.out.println("Loaded google model");

        System.out.println("Evaluate model....");
        double sim = wordVectors.similarity("stock", "outperform");
        System.out.println("Similarity between people and money: " + sim);
        Collection<String> similar = wordVectors.wordsNearest("stock", 10);
        System.out.println("Similar words to 'stock' : " + similar);
    }
}
