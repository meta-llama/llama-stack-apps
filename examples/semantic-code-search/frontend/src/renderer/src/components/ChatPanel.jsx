import React, { useState, useRef, useEffect } from 'react';
import { Box, TextField, IconButton, Paper, Typography, Tooltip } from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';
import OpenInNewIcon from '@mui/icons-material/OpenInNew';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import { Light as SyntaxHighlighter } from 'react-syntax-highlighter';
import js from 'react-syntax-highlighter/dist/esm/languages/hljs/javascript';
import python from 'react-syntax-highlighter/dist/esm/languages/hljs/python';
import { docco } from 'react-syntax-highlighter/dist/esm/styles/hljs';

// Register languages for syntax highlighting
SyntaxHighlighter.registerLanguage('javascript', js);
SyntaxHighlighter.registerLanguage('python', python);

const Message = ({ content, isUser, metadata }) => {
  const handleCopyCode = (code) => {
    navigator.clipboard.writeText(code);
  };

  const renderContent = () => {
    if (!content.includes('```')) {
      return <Typography sx={{ whiteSpace: 'pre-wrap' }}>{content}</Typography>;
    }

    const parts = content.split(/(```[a-z]*\n[\s\S]*?\n```)/);
    return parts.map((part, index) => {
      if (part.startsWith('```')) {
        const language = part.match(/```([a-z]*)\n/)?.[1] || 'text';
        const code = part.replace(/```[a-z]*\n/, '').replace(/\n```$/, '');
        
        return (
          <Box key={index} sx={{ my: 1, width: '100%' }}>
            <Paper 
              elevation={1}
              sx={{ 
                backgroundColor: '#ffffff',
                borderRadius: 1,
                position: 'relative',
                border: '1px solid #e0e0e0'
              }}
            >
              <Box sx={{ 
                position: 'absolute', 
                right: 8, 
                top: 8, 
                zIndex: 1,
                display: 'flex',
                gap: 0.5,
                backgroundColor: 'rgba(255, 255, 255, 0.9)',
                padding: '4px',
                borderRadius: 1,
                boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
              }}>
                <Tooltip title="Copy Code">
                  <IconButton 
                    size="small" 
                    onClick={() => handleCopyCode(code)}
                  >
                    <ContentCopyIcon fontSize="small" />
                  </IconButton>
                </Tooltip>
              </Box>
              <Box sx={{ p: 2 }}>
                <Typography 
                  variant="caption" 
                  sx={{ 
                    display: 'block',
                    mb: 1,
                    color: 'text.secondary',
                    fontFamily: 'monospace'
                  }}
                >
                  File: {metadata?.file_path || metadata?.name || 'unknown'}
                </Typography>
                <SyntaxHighlighter
                  language={language}
                  style={docco}
                  customStyle={{ 
                    margin: 0,
                    backgroundColor: 'transparent',
                    fontSize: '14px',
                    lineHeight: '1.5',
                  }}
                  showLineNumbers={true}
                  wrapLines={true}
                  wrapLongLines={true}
                >
                  {code}
                </SyntaxHighlighter>
              </Box>
            </Paper>
          </Box>
        );
      }
      return <Typography key={index} sx={{ whiteSpace: 'pre-wrap' }}>{part}</Typography>;
    });
  };

  return (
    <Box 
      sx={{ 
        display: 'flex',
        justifyContent: isUser ? 'flex-end' : 'flex-start',
        px: 2,
        py: 1
      }}
    >
      <Box 
        sx={{ 
          maxWidth: '85%',
          backgroundColor: isUser ? 'primary.main' : 'background.paper',
          color: isUser ? 'primary.contrastText' : 'text.primary',
          borderRadius: 2,
          p: 2,
          boxShadow: 1
        }}
      >
        {renderContent()}
      </Box>
    </Box>
  );
};

const ChatPanel = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const formatCodeResponse = (result) => {
    const { file, code, similarity } = result;
    return `Here's a relevant code snippet (similarity: ${similarity}):\n\`\`\`python\n${code}\n\`\`\``;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = {
      content: input,
      isUser: true
    };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch('http://localhost:5000/api/search', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: input
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to fetch results');
      }

      const data = await response.json();
      
      if (data.results && data.results.length > 0) {
        const aiMessage = {
          content: formatCodeResponse(data.results[0]),
          isUser: false,
          metadata: { file_path: data.results[0].file }
        };
        setMessages(prev => [...prev, aiMessage]);
      } else {
        setMessages(prev => [...prev, {
          content: "I couldn't find any relevant code snippets for your query.",
          isUser: false
        }]);
      }
    } catch (error) {
      console.error('Error fetching results:', error);
      setMessages(prev => [...prev, {
        content: "Sorry, I encountered an error while searching for code snippets. Please try again.",
        isUser: false
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Box sx={{ 
      height: '100%',
      width: '100%',
      display: 'flex',
      flexDirection: 'column',
      overflow: 'hidden'
    }}>
      <Box sx={{ 
        flex: 1,
        overflow: 'auto',
        '&::-webkit-scrollbar': {
          width: '8px',
        },
        '&::-webkit-scrollbar-track': {
          background: '#f5f5f5',
        },
        '&::-webkit-scrollbar-thumb': {
          background: '#ddd',
          borderRadius: '4px',
        },
        '&::-webkit-scrollbar-thumb:hover': {
          background: '#ccc',
        }
      }}>
        {messages.map((message, index) => (
          <Message key={index} {...message} />
        ))}
        <div ref={messagesEndRef} />
      </Box>
      
      <Paper
        component="form"
        onSubmit={handleSubmit}
        elevation={3}
        square
        sx={{
          p: 2,
          borderTop: 1,
          borderColor: 'divider',
          backgroundColor: 'background.paper',
          position: 'relative',
          zIndex: 1
        }}
      >
        <Box sx={{ display: 'flex', gap: 1 }}>
          <TextField
            fullWidth
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type your query here..."
            variant="outlined"
            size="small"
            disabled={isLoading}
            sx={{
              '& .MuiOutlinedInput-root': {
                backgroundColor: 'background.paper'
              }
            }}
          />
          <IconButton 
            type="submit" 
            color="primary" 
            disabled={isLoading}
            sx={{
              backgroundColor: theme => theme.palette.primary.main,
              color: 'white',
              '&:hover': {
                backgroundColor: theme => theme.palette.primary.dark,
              },
              '&.Mui-disabled': {
                backgroundColor: 'action.disabledBackground',
                color: 'white'
              }
            }}
          >
            <SendIcon />
          </IconButton>
        </Box>
      </Paper>
    </Box>
  );
};

export default ChatPanel;
