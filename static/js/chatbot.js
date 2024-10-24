$(document).ready(function () {
    let isTyping = false;

    // Toggle chatbot visibility on click
    $('#chatbot-icon').click(function () {
        $('#chat-container').toggleClass('show');
    });

    // Close chat
    $('#close-chat').click(function () {
        $('#chat-container').removeClass('show');
    });

    // Load frequently asked questions from the server
    loadFAQs();

    // Automatically send the message when Enter is pressed
    $('#user_input').on('keypress', function (e) {
        if (e.which == 13) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Disable send button if input is empty
    $('#user_input').on('input', function () {
        $('#send-button').prop('disabled', $(this).val().trim() === "");
    });

    // Handle clicking the send button
    $('#send-button').click(function () {
        sendMessage();
    });

    // Load FAQ and append to the chat
    function loadFAQs() {
        $.ajax({
            url: '/get_faqs',
            type: 'GET',
            success: function (response) {
                response.forEach(function (question) {
                    $('#faq-list').append('<li>' + question + '</li>');
                });

                // Add click event to FAQ items
                $('#faq-list li').click(function () {
                    var selectedFAQ = $(this).text();
                    $('#user_input').val(selectedFAQ);
                    sendMessage(); // Automatically send the message when FAQ is clicked
                });
            },
            error: function () {
                $('#faq-list').append('<li>Could not load FAQs. Please try again later.</li>');
            }
        });
    }

    // Handle sending a message
    function sendMessage() {
        var userInput = $('#user_input').val().trim();
        if (userInput === "") return;

        // Append user message to chatbox
        $('#chatbox').append('<div class="message user">' + userInput + '</div>');
        $('#user_input').val(''); // Clear input field
        $('#send-button').prop('disabled', true); // Disable send button after sending

        // Scroll chatbox to the bottom smoothly
        scrollChatbox();

        // Show typing indicator with delay for realism
        showTypingIndicator();

        // Simulate a delay in the bot's response
        setTimeout(function () {
            // Send the user message to the server and handle the response
            $.ajax({
                url: '/get_response',
                type: 'POST',
                contentType: 'application/json',
                data: JSON.stringify({ 'message': userInput }),
                success: function (response) {
                    hideTypingIndicator();

                    // Append bot response to chatbox
                    $('#chatbox').append('<div class="message bot">' + response.response + '</div>');
                    scrollChatbox(); // Smooth scroll
                    autoFocusInput(); // Auto-focus input field
                },
                error: function () {
                    hideTypingIndicator();

                    // Show error message
                    $('#chatbox').append('<div class="message bot">Sorry, there was an error. Please try again.</div>');
                    scrollChatbox(); // Smooth scroll
                    autoFocusInput(); // Auto-focus input field
                }
            });
        }, 1000); // 1-second delay for more natural conversation flow
    }

    // Show typing indicator
    function showTypingIndicator() {
        isTyping = true;
        $('#typing-indicator').fadeIn(300); // Smooth fade-in
    }

    // Hide typing indicator
    function hideTypingIndicator() {
        isTyping = false;
        $('#typing-indicator').fadeOut(300); // Smooth fade-out
    }

    // Smoothly scroll chatbox to the bottom
    function scrollChatbox() {
        $('#chatbox').animate({
            scrollTop: $('#chatbox')[0].scrollHeight
        }, 400);
    }

    // Auto-focus input after sending a message
    function autoFocusInput() {
        setTimeout(function () {
            $('#user_input').focus();
        }, 500); // Small delay before focusing input again
    }
});
