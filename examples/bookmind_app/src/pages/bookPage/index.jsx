import React, { useState } from "react";
import { FaSearch, FaBook, FaHome } from "react-icons/fa";
import { Link } from "react-router-dom";
import AISearch from "./components/AISearch";
import CharacterGraph from "./components/CharacterGraph";
import axios from "axios";

export default function BookPage() {
  const [searchTerm, setSearchTerm] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [graphData, setGraphData] = useState(null);
  const [bookData, setBookData] = useState(null);
  const [searchComplete, setSearchComplete] = useState(false);

  const verificationGraphData = (graphData) => {
    try {
      // Create Set of valid node IDs
      const nodeIds = new Set(graphData.nodes.map((node) => node.id));

      // Filter links to only include valid node references
      const validLinks = graphData.links.filter(
        (link) => nodeIds.has(link.source) && nodeIds.has(link.target)
      );

      return {
        nodes: graphData.nodes,
        links: validLinks,
      };
    } catch (error) {
      console.error("Error validating graph data:", error);
      return graphData; // Return original data if validation fails
    }
  };

  const initializeMemory = async (title) => {
    setIsLoading(true);
    try {
      const response = await axios.post("http://localhost:5001/initialize", {
        title: title,
      });

      // Verify and set graph data
      const verifiedData = verificationGraphData(response.data);
      setGraphData(verifiedData);

      return verifiedData;
    } catch (error) {
      console.error("Initialization error:", error);
      throw error;
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!searchTerm.trim()) {
      return;
    }

    setIsLoading(true);
    try {
      // Initialize memory and fetch book data in parallel
      const [memoryResponse, bookInfo] = await Promise.all([
        initializeMemory(searchTerm),
        fetchBookData(searchTerm),
      ]);

      setBookData({
        title: bookInfo.title,
        subtitle: bookInfo.summary,
        posterUrl: bookInfo.coverUrl,
        author: bookInfo.author,
        publishedDate: bookInfo.publishedDate,
        pageCount: bookInfo.pageCount,
      });

      setGraphData(memoryResponse);
      setSearchComplete(true);
    } catch (error) {
      console.error("Search error:", error);
    } finally {
      setIsLoading(false);
    }
  };
  // Add new function to fetch book cover
  const fetchBookData = async (bookTitle) => {
    try {
      const response = await axios.get(
        `https://www.googleapis.com/books/v1/volumes`,
        {
          params: {
            q: bookTitle,
            key: process.env.REACT_APP_GOOGLE_BOOKS_API_KEY,
          },
        }
      );

      if (response.data.items && response.data.items[0]) {
        const volumeInfo = response.data.items[0].volumeInfo;
        const imageLinks = volumeInfo.imageLinks || {};

        return {
          coverUrl:
            imageLinks.extraLarge ||
            imageLinks.large ||
            imageLinks.medium ||
            imageLinks.thumbnail ||
            "/placeholder.jpg",
          summary: volumeInfo.description || "No summary available",
          title: volumeInfo.title,
          author: volumeInfo.authors?.[0] || "Unknown Author",
          publishedDate: volumeInfo.publishedDate,
          pageCount: volumeInfo.pageCount,
        };
      }

      return {
        coverUrl: "/placeholder.jpg",
        summary: "No summary available",
        title: bookTitle,
        author: "Unknown Author",
        publishedDate: "",
        pageCount: 0,
      };
    } catch (error) {
      console.error("Error fetching book data:", error);
      return {
        coverUrl: "/placeholder.jpg",
        summary: "Failed to load book information",
        title: bookTitle,
        author: "Unknown Author",
        publishedDate: "",
        pageCount: 0,
      };
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 py-16 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        {/* Home Button */}
        <div className="flex justify-center mb-4">
          <Link
            to="/"
            className="flex items-center text-blue-600 hover:text-blue-800 transition-colors"
          >
            <FaHome className="mr-2" />
            Home
          </Link>
        </div>

        {/* Search Section */}
        <div className="max-w-md mx-auto mb-16">
          <h1 className="text-4xl font-extrabold text-center mb-8 text-gray-800 tracking-tight">
            <span className="bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-indigo-600">
              Character Mind Map
            </span>
          </h1>
          <div className="bg-white/80 backdrop-blur-sm shadow-xl rounded-xl p-8 space-y-6 transform transition-all duration-300 hover:scale-[1.02]">
            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="relative">
                <input
                  type="text"
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  placeholder="Enter book or movie title..."
                  className="w-full px-5 py-3 rounded-lg border-2 border-gray-200 
                         focus:border-blue-500 focus:ring-2 focus:ring-blue-200 
                         transition-all duration-200 bg-white/90
                         placeholder-gray-400 text-gray-700"
                  disabled={isLoading}
                />
              </div>
              <button
                type="submit"
                disabled={isLoading}
                className="w-full bg-gradient-to-r from-blue-500 to-indigo-600 
                       text-white font-semibold py-3 px-6 rounded-lg
                       transform transition-all duration-200
                       hover:from-blue-600 hover:to-indigo-700
                       focus:ring-2 focus:ring-offset-2 focus:ring-blue-500
                       disabled:opacity-50 disabled:cursor-not-allowed
                       flex items-center justify-center space-x-2"
              >
                <FaSearch className={`${isLoading ? "animate-spin" : ""}`} />
                <span>{isLoading ? "Searching..." : "Search"}</span>
              </button>
            </form>
          </div>
          <p className="mt-4 text-center text-sm text-gray-600">
            Search for any book or movie to explore character relationships
          </p>
        </div>

        {/* Info Section - Only show when search is complete */}
        {searchComplete && bookData && (
          <div className="space-y-8">
            <div className="bg-white shadow-xl rounded-xl overflow-hidden">
              <div className="md:flex">
                <div className="md:flex-shrink-0">
                  <img
                    src={bookData.posterUrl}
                    alt={bookData.title}
                    className="h-48 w-full object-cover md:h-full md:w-48"
                  />
                </div>
                <div className="p-8">
                  <div className="flex items-center">
                    <FaBook className="text-blue-500 mr-2" />
                    <h1 className="text-3xl font-bold text-gray-800">
                      {bookData.title}
                    </h1>
                  </div>
                  <p className="mt-2 text-gray-600">{bookData.subtitle}</p>
                </div>
              </div>
            </div>

            <div className="grid md:grid-cols-2 gap-8">
              <div className="bg-white/80 backdrop-blur-sm shadow-xl rounded-xl p-6">
                <AISearch bookTitle={bookData.title} />
              </div>
              <div className="bg-white/80 backdrop-blur-sm shadow-xl rounded-xl p-6">
                <CharacterGraph graphData={graphData} />
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
