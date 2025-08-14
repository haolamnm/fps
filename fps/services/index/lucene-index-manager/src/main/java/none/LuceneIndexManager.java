package none;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Stream;
import java.util.zip.GZIPInputStream;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.Tokenizer;
import org.apache.lucene.analysis.core.WhitespaceTokenizer;
import org.apache.lucene.analysis.miscellaneous.DelimitedTermFrequencyTokenFilter;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.IndexWriterConfig.OpenMode;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.store.FSDirectory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import net.sourceforge.argparse4j.ArgumentParsers;
import net.sourceforge.argparse4j.impl.action.StoreTrueArgumentAction;
import net.sourceforge.argparse4j.inf.ArgumentParser;
import net.sourceforge.argparse4j.inf.MutuallyExclusiveGroup;
import net.sourceforge.argparse4j.inf.Namespace;
import net.sourceforge.argparse4j.inf.Subparser;
import net.sourceforge.argparse4j.inf.Subparsers;

public class LuceneIndexManager {
    private static final Logger logger = LoggerFactory.getLogger(LuceneIndexManager.class);

    protected static class TermFrequencyAnalyzer extends Analyzer {
        @Override
        protected TokenStreamComponents createComponents(String fieldName) {
            Tokenizer source = new WhitespaceTokenizer();
            TokenStream result = new DelimitedTermFrequencyTokenFilter(source);
            return new TokenStreamComponents(source, result);
        }
    }

    protected static final FieldType STORED_TEXT_FIELD_TYPE;
    static {
        STORED_TEXT_FIELD_TYPE = new FieldType(TextField.TYPE_STORED);
        STORED_TEXT_FIELD_TYPE.setIndexOptions(IndexOptions.DOCS);
        STORED_TEXT_FIELD_TYPE.setOmitNorms(true);
        STORED_TEXT_FIELD_TYPE.freeze();
    }

    protected static final FieldType SURROGATE_TEXT_FIELD_TYPE;
    static {
        SURROGATE_TEXT_FIELD_TYPE = new FieldType(TextField.TYPE_NOT_STORED);
        SURROGATE_TEXT_FIELD_TYPE.setIndexOptions(IndexOptions.DOCS_AND_FREQS);
        SURROGATE_TEXT_FIELD_TYPE.setStoreTermVectors(true);
    }

    public static void main(String[] args) throws IOException {
        ArgumentParser parser = ArgumentParsers.newFor("LuceneIndexManager").build().defaultHelp(true)
                .description("lucene index manager");
        parser.addArgument("index_dir").help("directory that stores the index files");

        Subparsers subparsers = parser.addSubparsers().help("sub-command help");
        Subparser addParser = subparsers.addParser("add").setDefault("command", "add")
                .help("add or replace documents in an index");
        addParser.addArgument("-f", "--force").action(new StoreTrueArgumentAction())
                .help("whether to replace existing document or skip insertion");
        addParser.addArgument("documents_file_template")
                .help("jsonl.gz file containing documents to be added; {video_id} will be replaced with the video id");

        MutuallyExclusiveGroup group = addParser.addMutuallyExclusiveGroup("video_ids_input").required(true);
        group.addArgument("--video-ids").help("id of the video to which input documents belong");
        group.addArgument("--video-ids-list-path").help("path to a file containing a list of video ids to be added");

        Namespace namespace = parser.parseArgsOrFail(args);

        String subcommand = namespace.getString("command");
        if (subcommand.equals("add")) {
            addVideoIds(namespace);
        }
    }

    public static void addVideoIds(Namespace namespace) throws IOException {
        String documentsFileTemplate = namespace.getString("documents_file_template");
        String videoIdsListPath = namespace.getString("video_ids_list_path");
        String videoIdsString = namespace.getString("video_ids");
        String indexDir = namespace.getString("index_dir");
        boolean force = namespace.getBoolean("force");

        // Open the index directory
        Path indexDirPath = Paths.get(indexDir, "");

        try (
                FSDirectory index = FSDirectory.open(indexDirPath);
                Stream<String> videoIds = videoIdsListPath != null
                        ? Files.lines(Paths.get(videoIdsListPath)).map(String::trim)
                        : Stream.of(videoIdsString)) {
            videoIds.forEach(videoId -> {
                try {
                    addVideoId(videoId, documentsFileTemplate.replace("{video_id}", videoId), index, force);
                } catch (IOException e) {
                    logger.error("Failed to add video ID {}: {}", videoId, e.getMessage(), e);
                }
            });
        }
    }

    private static void addVideoId(String videoId, String documentsFile, FSDirectory index, boolean force)
            throws IOException {
        Term videoIdTerm = new Term("videoId", videoId);

        if (!force && DirectoryReader.indexExists(index)) {
            IndexReader indexReader = DirectoryReader.open(index);
            IndexSearcher indexSearcher = new IndexSearcher(indexReader);

            Query query = new TermQuery(videoIdTerm);
            long hits = indexSearcher.search(query, 1).totalHits.value();
            if (hits > 0) {
                logger.info("Found {} documents for video {}: skipping", hits, videoId);
                return;
            }
        }

        // Configure the index writer
        Analyzer analyzer = new TermFrequencyAnalyzer();
        IndexWriterConfig config = new IndexWriterConfig(analyzer);
        config.setOpenMode(OpenMode.CREATE_OR_APPEND);

        // Documents file
        try (
                InputStream fileStream = new FileInputStream(documentsFile);
                InputStream gzipStream = new GZIPInputStream(fileStream);
                Reader decoder = new InputStreamReader(gzipStream, StandardCharsets.UTF_8);
                BufferedReader bufferedReader = new BufferedReader(decoder);
                Stream<String> lines = bufferedReader.lines();) {
            // Iterate over JSON objects
            Stream<Document> documents = lines
                    .map(line -> JsonParser.parseString(line).getAsJsonObject())
                    .map(LuceneIndexManager::createDocument);
            Iterable<Document> documentsWithProgressBar = documents::iterator;

            try (IndexWriter writer = new IndexWriter(index, config)) {
                writer.updateDocuments(videoIdTerm, documentsWithProgressBar);
            }
        }
    }

    private static Document createDocument(JsonObject jsonObject) {
        Document doc = new Document();
        jsonObject.entrySet().stream()
                .map(LuceneIndexManager::getLuceneField)
                .filter(Objects::nonNull)
                .forEach(doc::add);

        return doc;
    }

    private static Field getLuceneField(Map.Entry<String, JsonElement> field) {
        String fieldName = field.getKey();
        JsonElement fieldValue = field.getValue();

        switch (fieldName) {
            // Searchable fields
            case "imageId":
            case "videoId":
                return new Field(fieldName, fieldValue.getAsString(), STORED_TEXT_FIELD_TYPE);

            // Stored int fields
            case "startFrame":
            case "endFrame":
            case "middleFrame":
                return new StoredField(fieldName, fieldValue.getAsInt());

            // Stored double fields
            case "startTime":
            case "endTime":
            case "middleTime":
                return new StoredField(fieldName, fieldValue.getAsDouble());

            // Indexed STR fields
            case "text":
            case "objects":
            case "features":
            case "aladin":
            case "features_clip-openai-clip-vit-large-patch14_str":
            case "features_clip-laion-CLIP-ViT-H-14-laion2B-s32B-b79K_str":
                return new Field(fieldName, fieldValue.getAsString(), SURROGATE_TEXT_FIELD_TYPE);

            // Ignored fields
            case "":
                return null;

            // Stored strind fields
            default:
                return new StoredField(fieldName, fieldValue.getAsString());
        }
    }
}
