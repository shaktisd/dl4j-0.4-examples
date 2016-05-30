package org.deeplearning4j.examples;

import org.canova.api.util.ClassPathResource;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.LineSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.EndingPreProcessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Collection;

/**
 * Created by Shakti on 5/31/2016.
 */
public class MyExample {
    public static void main(String args[]) throws FileNotFoundException {
        System.out.println("Load data....");
        ClassPathResource resource = new ClassPathResource("raw_sentences.txt");
        SentenceIterator iter = new LineSentenceIterator(resource.getFile());
        iter.setPreProcessor(new SentencePreProcessor() {
            @Override
            public String preProcess(String sentence) {
                return sentence.toLowerCase();
            }
        });

        ClassPathResource resource2 = new ClassPathResource("myarticle.txt");
        SentenceIterator iter2 = new LineSentenceIterator(resource2.getFile());
        iter2.setPreProcessor(new SentencePreProcessor() {
            @Override
            public String preProcess(String sentence) {
                return sentence.toLowerCase();
            }
        });

        System.out.println("Tokenize data....");
        final EndingPreProcessor preProcessor = new EndingPreProcessor();
        TokenizerFactory tokenizer = new DefaultTokenizerFactory();
        tokenizer.setTokenPreProcessor(new TokenPreProcess() {
            @Override
            public String preProcess(String token) {
                token = token.toLowerCase();
                String base = preProcessor.preProcess(token);
                base = base.replaceAll("\\d", "d");
                if (base.endsWith("ly") || base.endsWith("ing"))
                    System.out.println();
                return base;
            }
        });

        int batchSize = 1000;
        int iterations = 3;
        int layerSize = 150;

        System.out.println("Build model....");
        Word2Vec vec = new Word2Vec.Builder()
            .batchSize(batchSize) //# words per minibatch.
            .minWordFrequency(5) //
            .useAdaGrad(false) //
            .layerSize(layerSize) // word feature vector size
            .iterations(iterations) // # iterations to train
            .learningRate(0.025) //
            .minLearningRate(1e-3) // learning rate decays wrt # words. floor learning
            .negativeSample(10) // sample size 10 words
            .iterate(iter) //
            .tokenizerFactory(tokenizer)
            .build();
        vec.fit();

        System.out.println("Evaluate model....");
        double sim = vec.similarity("people", "money");
        System.out.println("Similarity between people and money: " + sim);
        Collection<String> similar = vec.wordsNearest("day", 10);
        System.out.println("Similar words to 'day' : " + similar);

        //output: [night, week, year, game, season, during, office, until, -]
        Collection<String> lst3 = vec.wordsNearest("sachin", 10);
        System.out.println(lst3);


    }
}
