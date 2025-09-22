import { useState, useRef } from 'react';
import axios from 'axios';

const API_URL = 'http://localhost:8000'; // Your backend URL

function App() {
  const [file, setFile] = useState(null);
  const [query, setQuery] = useState('');
  const [response, setResponse] = useState('');
  const [sources, setSources] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [uploadedPdfInfo, setUploadedPdfInfo] = useState(null); // {id: '...', filename: '...'}

  const fileInputRef = useRef(null);

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
    setUploadedPdfInfo(null); // Reset when a new file is selected
    setResponse('');
    setSources([]);
    setError('');
  };

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a PDF file first.');
      return;
    }
    setIsLoading(true);
    setError('');
    setResponse('');
    setSources([]);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await axios.post(`${API_URL}/upload_pdf/`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setUploadedPdfInfo({ id: res.data.pdf_id, filename: res.data.filename });
      setError(''); // Clear previous error
      // Clear the file input
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
      setFile(null);

    } catch (err) {
      console.error("Upload error:", err);
      setError(err.response?.data?.detail || 'Failed to upload PDF.');
      setUploadedPdfInfo(null);
    } finally {
      setIsLoading(false);
    }
  };

  const handleQuery = async () => {
    if (!query.trim()) {
      setError('Please enter a query.');
      return;
    }
    if (!uploadedPdfInfo) {
      setError('Please upload a PDF first.');
      return;
    }

    setIsLoading(true);
    setError('');
    setResponse('');
    setSources([]);

    try {
      const res = await axios.post(`${API_URL}/query/`, {
        query: query,
        pdf_id: uploadedPdfInfo.id,
      });
      setResponse(res.data.answer);
      setSources(res.data.sources || []);
    } catch (err) {
      console.error("Query error:", err);
      setError(err.response?.data?.detail || 'Failed to get answer.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100 flex flex-col items-center p-4 sm:p-8">
      <div className="w-full max-w-2xl bg-gray-800 shadow-2xl rounded-lg p-6 sm:p-8">
        <h1 className="text-3xl sm:text-4xl font-bold text-center text-teal-400 mb-8">
          PDF Q&A with Gemini & Weaviate
        </h1>

        {/* File Upload Section */}
        <div className="mb-8 p-6 bg-gray-700 rounded-lg shadow-md">
          <h2 className="text-xl font-semibold mb-3 text-teal-300">1. Upload PDF</h2>
          <div className="flex flex-col sm:flex-row items-center space-y-3 sm:space-y-0 sm:space-x-3">
            <input
              type="file"
              accept=".pdf"
              onChange={handleFileChange}
              ref={fileInputRef}
              className="block w-full text-sm text-gray-300
                         file:mr-4 file:py-2 file:px-4
                         file:rounded-md file:border-0
                         file:text-sm file:font-semibold
                         file:bg-teal-500 file:text-white
                         hover:file:bg-teal-600 cursor-pointer"
            />
            <button
              onClick={handleUpload}
              disabled={isLoading || !file}
              className="w-full sm:w-auto px-6 py-2.5 bg-blue-600 text-white font-medium text-xs leading-tight uppercase rounded shadow-md hover:bg-blue-700 hover:shadow-lg focus:bg-blue-700 focus:shadow-lg focus:outline-none focus:ring-0 active:bg-blue-800 active:shadow-lg transition duration-150 ease-in-out disabled:opacity-50"
            >
              {isLoading && !uploadedPdfInfo ? 'Uploading...' : 'Upload PDF'}
            </button>
          </div>
          {uploadedPdfInfo && (
            <p className="mt-3 text-sm text-green-400">
              Uploaded: {uploadedPdfInfo.filename} (ID: {uploadedPdfInfo.id.substring(0,8)}...)
            </p>
          )}
        </div>

        {/* Query Section */}
        {uploadedPdfInfo && (
          <div className="mb-8 p-6 bg-gray-700 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-3 text-teal-300">2. Ask a Question</h2>
            <textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder={`Ask something about ${uploadedPdfInfo.filename}...`}
              rows="3"
              className="w-full p-3 bg-gray-800 border border-gray-600 rounded-md focus:ring-2 focus:ring-teal-500 focus:border-teal-500 placeholder-gray-500 text-gray-100"
            />
            <button
              onClick={handleQuery}
              disabled={isLoading || !query.trim()}
              className="mt-3 w-full px-6 py-2.5 bg-green-600 text-white font-medium text-xs leading-tight uppercase rounded shadow-md hover:bg-green-700 hover:shadow-lg focus:bg-green-700 focus:shadow-lg focus:outline-none focus:ring-0 active:bg-green-800 active:shadow-lg transition duration-150 ease-in-out disabled:opacity-50"
            >
              {isLoading && response === '' ? 'Thinking...' : 'Get Answer'}
            </button>
          </div>
        )}

        {/* Error Display */}
        {error && (
          <div className="mb-6 p-4 bg-red-700 text-white rounded-md">
            <p className="font-semibold">Error:</p>
            <p>{error}</p>
          </div>
        )}

        {/* Response Section */}
        {response && (
          <div className="p-6 bg-gray-700 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-3 text-teal-300">Answer:</h2>
            <p className="whitespace-pre-wrap text-gray-200 leading-relaxed">{response}</p>
            
            {sources && sources.length > 0 && (
              <div className="mt-6">
                <h3 className="text-lg font-semibold mb-2 text-teal-400">Sources (Retrieved Chunks):</h3>
                <ul className="list-disc list-inside space-y-2 max-h-60 overflow-y-auto p-3 bg-gray-800 rounded-md">
                  {sources.map((source, index) => (
                    <li key={index} className="text-sm text-gray-400">
                      <strong>Page {source.page_number || 'N/A'}:</strong> {source.text.substring(0, 150)}...
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
      </div>
      <footer className="text-center text-gray-500 mt-8 text-sm">
        PDF Q&A System | Backend: FastAPI, LLM: Gemini, VectorDB: Weaviate
      </footer>
    </div>
  );
}

export default App;