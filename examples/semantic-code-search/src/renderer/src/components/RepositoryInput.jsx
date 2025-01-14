import React, { useState } from 'react'
import PropTypes from 'prop-types'

const RepositoryInput = ({ onSubmit, isIndexing }) => {
  const [input, setInput] = useState('')
  const [error, setError] = useState('')

  const validateInput = (value) => {
    if (!value) return 'Please enter a GitHub URL or local path'
    
    // Validate GitHub URL
    if (value.startsWith('http')) {
      const githubPattern = /^https:\/\/github\.com\/[\w-]+\/[\w-]+/
      if (!githubPattern.test(value)) {
        return 'Invalid GitHub URL format. Expected: https://github.com/username/repo'
      }
    } else {
      // Validate local path (basic validation)
      if (!value.startsWith('/')) {
        return 'Local path must be absolute (start with /)'
      }
    }
    return ''
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    const validationError = validateInput(input)
    if (validationError) {
      setError(validationError)
      return
    }
    setError('')
    onSubmit(input)
  }

  return (
    <div className="max-w-2xl mx-auto p-6">
      <h2 className="text-2xl font-bold mb-6">Repository Input</h2>
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Enter GitHub URL or Local Path..."
            className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            disabled={isIndexing}
          />
          {error && <p className="text-red-500 mt-1 text-sm">{error}</p>}
        </div>
        <button
          type="submit"
          disabled={!input || isIndexing}
          className={`w-full py-3 px-4 rounded-lg text-white font-medium
            ${
              !input || isIndexing
                ? 'bg-gray-400 cursor-not-allowed'
                : 'bg-blue-600 hover:bg-blue-700'
            }`}
        >
          {isIndexing ? (
            <div className="flex items-center justify-center space-x-2">
              <div className="animate-spin rounded-full h-5 w-5 border-2 border-white border-t-transparent"></div>
              <span>Indexing Repository...</span>
            </div>
          ) : (
            'Index Repository'
          )}
        </button>
      </form>
    </div>
  )
}

RepositoryInput.propTypes = {
  onSubmit: PropTypes.func.isRequired,
  isIndexing: PropTypes.bool
}

RepositoryInput.defaultProps = {
  isIndexing: false
}

export default RepositoryInput
