/**
 * uif-main.js
 */

// hack for IE
if (typeof window.console === "undefined") {
    console = {
        log: function (msg) {
        }
    };
}

var needToConfirm = false;
var moneySettings = {aSign: "$", nBracket: "(,)", vMin: -9999999.99};
var numberSettings = {nBracket: "(,)", mDec: 0};

//show spinner during AJAX actions, hide it when action complete
if (typeof jQuery !== "undefined") {
    (function ($) {
        $(document).ajaxStart(function () {
            $("#spinner").fadeIn();
        }).ajaxStop(function () {
            $("#spinner").fadeOut();
        });
    })(jQuery);
}

window.onbeforeunload = confirmUnload;

function confirmUnload() {
    if (needToConfirm) {
        return "You are attempting to leave this page with unsaved changes.";
    }
}

function confirmExit(msg) {
    if (needToConfirm) {
        if (!confirm(msg)) {
            event.stopPropagation();
            event.preventDefault();
            return false;
        } else {
            needToConfirm = false;
            return true;
        }
    } else {
        return true;
    }
}

function showAjaxError(xhr) {
    if (xhr.statusText === "Internal Server Error") {
        $("#mainContent").html(
            "<h1>Sorry, an unexpected error occurred.</h1>" +
            "<p>Please be assured that information about it has been communicated to the administrators.</p>"
        );
    } else {
        $("#mainContent").html(xhr.statusText);
    }
}

function ajaxClick(event, data) {
    var link = event.currentTarget;
    if ($(link).attr("id") === "navigation-tree") {
        link = event.target;
        if ($(link).attr("href") === undefined || $(link).attr("href") === "#") {
            //event.preventDefault();
            return;
        }
    }

    var subprogramAcronym = $(link).attr("subprogramAcronym");
    if (subprogramAcronym) {
        $("#subprogramAcronym").val(subprogramAcronym);
    }

    //add timestamp to prevent IE caching of AJAX response - IE will always assume this call is different because the timestamp has changed
    var url = $(link).attr("href");
    var time = new Date().getTime();
    if (url) {
        if (url.indexOf("?") >= 0) {
            url = url + "&timestamp=" + time;
        } else {
            url = url + "?timestamp=" + time;
        }
    }
    $("#mainContent").load(url, function (response, status, xhr) {
        if (status === "error") {
            showAjaxError(xhr);
        } else {
            addEventHandling();
        }
    });

    event.preventDefault();
    event.stopPropagation();
    return false;
}

//all links in ajax-links class use ajax to load content into main content area
$(".ajax-links").on("click", "a:not(.no-ajax)", ajaxClick);


function searchResultsAccordion() {
    //Makes sure search tab is open, just in case user has navigated away
    $("#tabs").tabs("option", "active", 1);

    //make the results fold up like an accordion
    $(".accordion").accordion({
        header: "h3",
        heightStyle: "content",
        navigation: true,
        collapsible: true
    });

    //add search results message to advanced search popup
    var searchMessage = $("#searchMessage").val();
    $("#advancedSearchMessage").replaceWith("<div id='advancedSearchMessage' class='message' role='status'>" + searchMessage + "</div>");

    $("#advanced-search-buttons").find(":input").show();
}

function closeDialog(dialogId) {
    $("#" + dialogId).dialog("close");
    $("body").find(".ui-dialog-content").dialog("close");
}

function closeDialogIfNoErrors(dialogId) {
    var errors = $("#" + dialogId).find(".errors");
    if (errors.length === 0) {

        //change font on all comments tabs
        $("#tabsCompanyCommentsLink").removeClass("normal-font");
        $("#tabsPersonCommentsLink").removeClass("normal-font");
        $("#tabsFacilityCommentsLink").removeClass("normal-font");
        $("#tabsParkCommentsLink").removeClass("normal-font");
        $("#tabsUnitCommentsLink").removeClass("normal-font");
        $("#tabsAssignmentCommentsLink").removeClass("normal-font");
        $("#tabsComplaintCommentsLink").removeClass("normal-font");

        var updateDiv = $("#updateDiv").val();
        if (updateDiv && updateDiv === "ALL_COMMENTS") {
            $("#mainContent").load(editCommentsUrl, function (response, status, xhr) {
                if (status === "error") {
                    showAjaxError(xhr);
                } else {
                    addEventHandling();
                }
            });
        }

        $("#" + dialogId).dialog("close");
        $("body").find(".ui-dialog-content").dialog("close");
    }
}

function registerShortcutKey(linkId, keyCombination) {

    $(document).keydown(keyCombination, function (event) {
        //call click event on link and prevent the key event from bubbling up to the browser
        $(linkId).click();
        event.stopPropagation();
        event.preventDefault();
        return false;
    });
    //necessary for key event to fire when focus is in input elements
    $(":input").keydown(keyCombination, function (event) {
        $(linkId).click();
        event.stopPropagation();
        event.preventDefault();
        return false;
    });
}

function registerShortcutKeys() {
    var elementsWithShortcuts = $("input[shortcutkey]");
    $.each(elementsWithShortcuts, function (index, domElement) {
        registerShortcutKey("#" + $(domElement).attr("id"), $(domElement).attr("shortcutkey"));
    });

    var linksWithShortcuts = $("a[shortcutkey]");
    $.each(linksWithShortcuts, function (index, domElement) {
        registerShortcutKey("#" + $(domElement).attr("id"), $(domElement).attr("shortcutkey"));
    });
}

function formatDate(input) {
    var dateString = $(input).val();
    if (null != dateString) {
        if (dateString.length === 8) {
            dateString = dateString.substring(0, 2) + "/" + dateString.substring(2, 4) + "/" + dateString.substring(4, 8);
        }
        $(input).val(dateString);
    }
}

function formatPhone(input) {
    var phoneString = $(input).val();
    if (null != phoneString) {
        if (phoneString.length === 10) {
            phoneString = "(" + phoneString.substring(0, 3) + ") " + phoneString.substring(3, 6) + "-" + phoneString.substring(6, 10);
        }
        $(input).val(phoneString);
    }
}

function addAutoComplete(input, baseUrl) {
    if (baseUrl && input && input.attr("controller") && input.attr("action")) {
        input.autocomplete({
            source: function (request, response) {
                var $element = $(this.element);

                //get the previous ajax request stored in jqXHR variable
                var previous_request = $element.data("jqXHR");

                //abort previous request
                if (previous_request) {
                    previous_request.abort();
                }

                // Store new AJAX request in jqXHR variable
                $element.data("jqXHR", $.ajax({
                    url: baseUrl + input.attr("controller") + "/" + input.attr("action"),
                    data: request,
                    success: function (data) {
                        response(data); // set the response
                    }
                }));
            },
            minLength: 3,
            delay: 800, //milliseconds
            autoFocus: true,
            select: function (event, ui) {
                input.val(ui.item.value);
                if (input.attr("idField")) {
                    $("#" + input.attr("idField")).val(ui.item.id);
                }
            }
        });
    }
}

function addScrollTable() {
    //make table in a "scrollable-table" div scroll, but maintain the headers
    $("div.scrollable-table").tableScroll({
        flush: false,
        width: null,
        height: null
    });
}

function addTextAreaMaxSizeConstraint() {
    // ignore these keys
    var ignore = [8, 9, 13, 33, 34, 35, 36, 37, 38, 39, 40, 46];

    // handle textareas with maxlength attribute
    $("textarea[maxlength]")
        .bind("keypress", function (event) { //stop keypress if it will put length over maxlength
            var self = $(this);
            var maxlength = self.attr("maxlength");
            var code = $.data(this, "keycode");

            // check if maxlength has a value.
            // The value must be greater than 0
            if (maxlength && maxlength > 0) {

                // continue with this keystroke if maxlength
                // not reached or one of the ignored keys were pressed.
                return (self.val().length < maxlength
                    || $.inArray(code, ignore) !== -1);

            }
        })
        .bind("paste", function (event) { //stop paste event - crop text if over the limit
            var self = $(this);
            var maxlength = self.attr("maxlength");

            // check if maxlength has a value
            if (maxlength && maxlength > 0) {
                // set timeout so text has time to be added to the textarea,
                // otherwise self.val().length will be equal to the original length, not including the new text
                setTimeout(function () {
                    if (self.val().length > maxlength) {
                        //crop text down to maxlength
                        var croppedText = self.val().substring(0, maxlength - 1);
                        self.val(croppedText);
                    }
                }, 500);
            }
        })
        .off("keydown").on("keydown", function (event) {
            $.data(this, "keycode", event.keyCode || event.which);
        });
}

function addFormEvents(baseUrl) {
    registerShortcutKeys();

    /**
     * Display a warning on capslock
     */
    $('.caps-lock-warning').keypress(function(e) {
        var capsLockMessage = $('.caps-lock-message');
        // Detect current character & shift key
        var character = e.keyCode ? e.keyCode : e.which;
        var sftKey = e.shiftKey ? e.shiftKey : ((character == 16) ? true : false);
        isCapsLock = (((character >= 65 && character <= 90) && !sftKey) || ((character >= 97 && character <= 122) && sftKey));
        if (isCapsLock === true) {
            capsLockMessage.show();
        } else {
            capsLockMessage.hide();
        }
    });

    $("form.enter-key-control").off("keypress").on("keypress", function(event){
        if(event.which == 13 ) {
            var target = $( event.target );
            if(target.is('textarea')) {
                //ok
                return true;
            } else {
                var buttonToPress = $(this).find('.enter-key-button');
                if(buttonToPress) {
                    buttonToPress.click();
                }
                event.preventDefault();
                return false;
            }
        }
    });

    //adds a popup calendar to all date fields
    $(".date-field:enabled").datepicker({
        showAnim: false,
        duration: 0,
        changeMonth: true,
        changeYear: true,
        closeText: "X",
        constrainInput: true,
        showButtonPanel: true,
        showOn: "button",
        buttonText: "Calendar",
        buttonImage: $("#cookedVars").data("calendar-image"),
        buttonImageOnly: true,
        onSelect: function () {
            this.focus();
        }
    });

    $(".date-field").on("blur change", function (e) {

        var dateValue = $(this).val();
        if (dateValue == "0") {
            $(this).val($.datepicker.formatDate("mm/dd/yy", new Date()));
        } else if (dateValue.length === 6 && !isNaN(dateValue)) {
            var month = dateValue.substring(0, 2);
            var day = dateValue.substring(2, 4);
            var year = dateValue.substring(4);

            //per John, pivoting on 1940 - 40 goes in as 1940, 39 goes in as 2039
            if (year < 40) {
                year = "20" + year;
            } else {
                year = "19" + year;
            }

            var dateString = month + "/" + day + "/" + year;

            var dateEntered = new Date(dateString);
            $(this).val($.datepicker.formatDate("mm/dd/yy", dateEntered));
        }

        formatDate($(this));

        if ($(this).hasClass("no-future-date")) {
            isFutureDate($(this), null);
        }

        //need to put this here otherwise
        if ($(this).hasClass("new-unit")) {
            checkDateBeforeDateWarning($("#unitFirstSoldDate"), "First Sold Date", $("#unitManufacturedDate"), "Manufactured Date");
        }

        return true;
    });

    // special date keys "+" and "-" increment or decrement the date value
    $(".date-field").keydown(function (e) {
        var addDays = 0;
        var dateValue = null;
        var dateString = null;

        if (e.key == "-") {
            addDays = -1; // decrement date
        } else if (e.key == "+") {
            addDays = 1;
        }
        if (addDays !== 0) {

            dateString = $(this).val();
            if (dateString.trim().length === 10) {
                // probably a date
                dateValue = new Date(dateString);
                if (!isNaN(dateValue.getTime())) {

                    // it is a date, do the crement
                    dateValue = new Date(dateValue.getFullYear(),
                        dateValue.getMonth(),
                        dateValue.getDate() + addDays);

                    // stuff it back in to the input
                    var mm = ("0" + (dateValue.getMonth() + 1)).slice(-2);
                    var dd = ("0" + (dateValue.getDate())).slice(-2);
                    var yyyy = ("" + (dateValue.getFullYear()));
                    dateString = mm + "/" + dd + "/" + yyyy;
                    $(e.currentTarget).val(dateString);

                    // don't add this special character
                    e.preventDefault();
                }
            } // else not a date, go ahead with the regular stuff
        }
    });

    $(".phone-field").blur(function () {
        formatPhone($(this));
    });

    $(".autocomplete-field").each(function () {
        addAutoComplete($(this), baseUrl);
    });

    // adds number formatting and functions for doing math with formatted values
    $(".money").autoNumeric("init", moneySettings);
    $(".number").autoNumeric("init", numberSettings);

    // allows Enter key on filter inputs to submit filter criteria
    // must add id="list-filter" clause to filter fieldset and class="apply-button" to apply action:
    // <fieldset class="box" id="list-filter">
    // <g:render template="/common/filterButtons" />
    $("#list-filter").off("keypress").on("keypress", "input", function (e) {
        if (e.which === 13) { // Enter key
            event.stopPropagation();
            event.preventDefault();
            // submit the filter request
            $("#list-filter .apply-button").click();
        }
    });

    //allow select boxes with this class to submit the surrounding form when a new value is selected
    $(".ajax-form-select").change(function () {
        var form = $(this).closest("form");
        var method = $(form).attr("method");
        var url = $(form).attr("action");

        //prevents IE from caching Ajax call
        if (url) {
            if (url.indexOf("?") >= 0) {
                url = url + "&timestamp=" + new Date().getTime();
            } else {
                url = url + "?timestamp=" + new Date().getTime();
            }

            $.ajax({
                type: method,
                url: url,
                data: $(form).autoNumeric("getString"),
                success: function (data) {
                    $("#mainContent").html(data);
                    addFormEvents(baseUrl);
                }
            });
        }

        return false;

    });

    $(".ajax-upload-button").off("click").on("click", function () {
        $('#uploadInProgressMessage').show();
        $('#errorMessages').empty();

        this.disabled=true;
        this.value='Sending…';

        var form = $(this).closest("form");
        var url = $(form).attr("action");
        $(".ajax-upload-fields").upload(url, function (data) {
            $("#mainContent").html(data);
            addFormEvents(baseUrl);
        }, "html");
    });

    addScrollTable();
    addTextAreaMaxSizeConstraint();
    needToConfirm = false;

    $(".confirm-request").off("click").on("click", function (e) {
        if (needToConfirm) {
            var msg = $(this).attr("message");
            if (!msg) msg = "You are attempting to leave this page with unsaved changes.  Are you sure you want to exit this page?";
            if (ifConfirm(msg,
                function () {
                    ajaxClick(e, null);
                },
                function () {
                    e.stopPropagation();
                    e.preventDefault();
                    return false;
                }, null)) {
                return true;
            } else {
                e.stopPropagation();
                e.preventDefault();
                return false;
            }
        } else {
            return true;
        }
    });

    $(".confirm-request-always").off("click").on("click", function (e) {
        var msg = $(this).attr("message");
        if (!msg) msg = "Are you sure?";
        if (ifConfirm(msg,
            function () {
                ajaxClick(e, null);
            },
            function () {
                e.stopPropagation();
                e.preventDefault();
                return false;
            }, null)) {
            return true;
        } else {
            e.stopPropagation();
            e.preventDefault();
            return false;
        }
    });

    //prevent backspace key from navigating back to login screen
    $(function () {
        /*
         * this swallows backspace keys on any non-input element.
         * stops backspace -> back
         */
        var rx = /INPUT|TEXTAREA/i;

        $(document).bind("keydown keypress", function (e) {
            if (e.which === 8) { // 8 == backspace
                if (!rx.test(e.target.tagName) || e.target.disabled || e.target.readOnly) {
                    e.preventDefault();
                }
            }
        });
    });


    var dialogButton = $(".ui-dialog-buttonset button:eq(0)");
    if (dialogButton.attr("class")) { //always returning true unless looking for something else like class
        dialogButton.focus();
    } else {
        moveToFirstFocus();
    }

    //Removes asterisk from required inputs that are disabled or readonly
    $("#mainContent :input:disabled").prev("label").children(".required-indicator").html("");
    $("#mainContent :input[readonly='readonly']").prev("label").children(".required-indicator").html("");

    $("#mainContent :input:disabled").parent("span").prev("label").children(".required-indicator").html("");
    $("#mainContent :input[readonly='readonly']").parent("span").prev("label").children(".required-indicator").html("");
}

//keeps the left navigation resized with the browser window
function resizeUi() {
    var height = $(window).height() - ($("#header").height() + $("#footer").height() + 35);
    var headerHeight = $("#header").height() + 10;
    $("#leftNavMaximized").height(height + 5);
    //$("#leftNavMaximized").css(top, headerHeight);
    $("#tabs").height(height);
    //$("#tabs").css(top, headerHeight);
    $("#leftNavMinimized").height(height);
    //$("#leftNavMinimized").css(top, headerHeight);

    var contentWidth = $(this).width() - 35;
    if ($("#leftNavMinimized:visible").length > 0) {
        contentWidth = contentWidth - ($("#leftNavMinimized").width());
    } else {
        contentWidth = contentWidth - ($("#leftNavMaximized").width());
    }

    $("#mainContent").width(contentWidth);
    $("#mainContent").height(height - 5);

}

var $mainDialog = $("#mainDialog").dialog({
    autoOpen: false,
    closeOnEscape: true,
    title: "CASAS 2 Message",
    minHeight: 300,
    minWidth: 400,
    modal: false,
    buttons: [
        {
            text: "Close",
            click: function () {
                $(this).dialog("close");
            }
        }
    ]
});

function openDialog(title, message) {
    if (title && message) {
        if (title != "default") {
            $mainDialog.dialog("option", "title", title);
        }

        $mainDialog.html(message);
        $mainDialog.dialog("open");

        return false;
    } else {
        return true;
    }
}

$(document).ready(function () {

    //only show main area after page load so user doesn't see it in a disorganized state
    $("#loadingMask").hide();
    $(".site-container").show();

    //adds tabs to left navigation
    $("#tabs").tabs();

    //bind enter key to simple search return
    $("#searchString").bind("keydown", function (e) {
        var code = e.keyCode || e.which;
        if (code === 13 || code === 9) { //Enter (13) keycode, tab (9) keycode
            e.preventDefault();
            $(".simpleSearchButton").click();
            //Do something
        }
    });
    $("#refNumberSearchString").bind("keydown", function (e) {
        var code = e.keyCode || e.which;
        if (code === 13 || code === 9) { //Enter (13) keycode, tab (9) keycode
            e.preventDefault();
            $(".refNumSimpleSearchButton").click();
            //Do something
        }
    });
    $("#searchStringAssignment").bind("keydown", function (e) {
        var code = e.keyCode || e.which;
        if (code === 13 || code === 9) { //Enter (13) keycode, tab (9) keycode
            e.preventDefault();
            $(".assignmentSearchButton").click();
            //Do something
        }
    });

    $("#searchStringComplaint").bind("keydown", function (e) {
        var code = e.keyCode || e.which;
        if (code === 13 || code === 9) { //Enter (13) keycode, tab (9) keycode
            e.preventDefault();
            $(".complaintSearchButton").click();
            //Do something
        }
    });

    $("#personCompanyIdSearchString").bind("keydown", function (e) {
        var code = e.keyCode || e.which;
        if (code === 13 || code === 9) { //Enter (13) keycode, tab (9) keycode
            e.preventDefault();
            $(".idSimpleSearchButton").click();
            //Do something
        }
    });
    $("#parkIdSearchString").bind("keydown", function (e) {
        var code = e.keyCode || e.which;
        if (code === 13 || code === 9) { //Enter (13) keycode, tab (9) keycode
            e.preventDefault();
            $(".idSimpleSearchButton").click();
            //Do something
        }
    });
    $("#facilityIdSearchString").bind("keydown", function (e) {
        var code = e.keyCode || e.which;
        if (code === 13 || code === 9) { //Enter (13) keycode, tab (9) keycode
            e.preventDefault();
            $(".idSimpleSearchButton").click();
            //Do something
        }
    });
    $("#unitIdSearchString").bind("keydown", function (e) {
        var code = e.keyCode || e.which;
        if (code === 13 || code === 9) { //Enter (13) keycode, tab (9) keycode
            e.preventDefault();
            $(".idSimpleSearchButton").click();
            //Do something
        }
    });

    $("#homeSearchString").bind("keydown", function (e) {
        var code = e.keyCode || e.which;
        if (code === 13 || code === 9) { //Enter (13) keycode, tab (9) keycode
            e.preventDefault();
            $(".idHomeSearchButton").click();
            //Do something
        }
    });

    //makes tab div resizable
    $("#leftNavMaximized").resizable({
        handles: "e",
        alsoResize: "#tabs",
        maxWidth: 600,
        minWidth: 100
    });

    //collapses and expands the left navigation
    $(".collapse-menu-btn").off("click").on("click", function (event) {
        $("#leftNavMaximized").hide();
        $("#leftNavMinimized").show();
        $(window).resize();

        // prevent the default action, e.g., following a link
        event.preventDefault();
        event.stopPropagation();
        return false;
    });

    $(".expand-menu-btn").off("click").on("click", function (event) {
        $("#leftNavMinimized").hide();
        $("#leftNavMaximized").show({
            effect: "fold" //try slide, drop, or fold
        });
        $(window).resize();

        // prevent the default action, e.g., following a link
        event.preventDefault();
        event.stopPropagation();
        return false;
    });

    //override browsers F1 Help popup so we can replace F1 with our own online help
    document.onhelp = new Function("return false;");
    window.onhelp = new Function("return false;");

    registerShortcutKeys();

    //makes sure event only fires when user finishes resizing
    var timeOut = false;
    $(window).resize(function () {
        if (timeOut !== false)
            clearTimeout(timeOut);
        timeOut = setTimeout(resizeUi, 200); //200 is time in milliseconds
    });
    $(window).resize();

    // Wire up the expand/collapse all buttons to their trees
    $(document).ready(function () {
        $(".tree-control").on("click", ".expand-all", function () {
            $(this).closest(".tree-control").next(".jstree").jstree("open_all");
        });
        $(".tree-control").on("click", ".collapse-all", function () {
            $(this).closest(".tree-control").next(".jstree").jstree("close_all");
        });

        // wire up the tab widget to tab divs
//        $("[id$=-tabs]").tabs();

    });

    // open/close submenu when the value is clicked
    $("#navigation-tree").on("click", ".submenu", function (event, data) {
        return $("#navigation-tree").jstree("toggle_node", $(event.currentTarget).closest("li"));
    });
    $("#navigation-tree").on("click", "li.menu-item a", function (event, obj) {
        if (needToConfirm) {

            event.preventDefault();
//            event.stopPropagation();

            ifConfirm($(this).attr("msg"),
                function () {     // yes
                    needToConfirm = false;
                    ajaxClick(event, obj);
                },
                function () {     // cancel
                    return;
                }
            );
        } else {
            ajaxClick(event, obj);
        }
    });

    $("#rolesLink").qtip(
        {
            content: {
                title: $("#rolesTitle"),
                text: $("#rolesContent")
            },
            position: {
                my: "top left",  // Position my top left...
                at: "bottom left"
            },
            style: {
                widget: true, // don't use the jQuery UI widget classes
                def: true
            },
            show: {
                delay: 1000,
                effect: function () {
                    $(this).slideDown();
                }
            },
            hide: {
                effect: function () {
                    $(this).slideUp();
                }
            }
        }
    );
});

function confirmLeavePage(msg) {
    if (needToConfirm) {
        if (!confirm(msg)) {
            return false;
        } else {
            needToConfirm = false;
            return true;
        }
    } else {
        return true;
    }
}

/**
 * Perform the given function when a user confirms they want to.
 * Allows for behavior on cancel, as well.
 *
 * Assumes an event triggered this call and sends the event's current target to the given functions
 *.
 * @param question  optional  'Are you sure?' is the default, if a function, the result will be used as the prompt
 * @param yesFn
 * @param cancelFn
 * @param finallyFn
 */
function ifConfirm(question, yesFn, cancelFn, finallyFn) {

    var title;
    try {
        var mainOpts = $("#mainDialog").dialog("option");
        title = mainOpts.title;
    } catch (ignore) {
        title = ($("#cookedVars").data("confirm-title"))
            ? $("#cookedVars").data("confirm-title")
            : "Confirm";
    }

    var question = (question) ? question : "Are you sure?";
    // hack around IE which does not have event.currentTarget
    var originatingElement = (event.currentTarget === undefined) ? event.srcElement : $(event.currentTarget);

    $("<div></div>")
        .html(question)
        .dialog({
            autoOpen: true,
            modal: true,
            title: title,
            minWidth: 350,
            minHeight: 250,
            buttons: {
                "Yes": function () {
                    // execute the yes function
                    if (typeof yesFn === "function") yesFn(event, originatingElement);
                    $(this).dialog("close");
                },
                "Cancel": function () {
                    if (typeof cancelFn === "function") cancelFn(event, originatingElement);
                    $(this).dialog("close");
                }
            },
            close: function () {
                if (typeof finallyFn === "function") finallyFn(event, originatingElement);
                $(this).remove();
            }
        });
}

function moveToFirstFocus() {
    var firstFocusElement = $("#mainContent").find(".first-focus");
    if (firstFocusElement.length && firstFocusElement.is(":visible")) {
        //puts focus on element with class first-focus
        firstFocusElement.focus();
    } else {
        //puts focus on first enabled input, select, textarea, or link in the content div
        $("#mainContent :input:visible:enabled:not([readonly='readonly']):first").focus();
    }
}

/**
 * alert Dialog
 * @param message  REQUIRED  the message for the user
 * @param title    optional  the title of the message pop-up. default is 'Notice'
 */
function alertUser(message, title) {
    $("<div></div>")
        .html(message)
        .dialog({
            autoOpen: true,
            modal: true,
            title: title ? title : "Notice",
            buttons: {
                "OK": function () {
                    $(this).dialog("close");
                }
            },
            close: function () {
                $(this).remove();
            }
        });
}

function isFutureDate(fieldId, fieldDisplayName) {
    var checkDate = fieldId.val();
    var label = fieldDisplayName;

    if (checkDate) {
        checkDate = checkDate.split("/");
        var today = new Date().getTime();
        checkDate = new Date(checkDate[2], checkDate[0] - 1, checkDate[1]).getTime();
        if ((today - checkDate) < 0) {
            if (!label) {   // use the label text for the date entry input
                label = $(fieldId).prev("label, span.label").text();
                if (!label) {
                    label = "Date";
                } else {
                    label = label.replace("*", "");
                }
            }
            fieldId.val("");
            createRedAlert("Validation Error", label + " cannot be in the future");
            return false;
        } else {
            return true;
        }
    }
}

function checkDateBeforeDateWarning(field1, fieldDisplayName1, field2, fieldDisplayName2) {
    if (field1 && field2) {
        var checkDate1 = field1.val();
        var checkDate2 = field2.val();

        if (checkDate1 && checkDate2) {
            checkDate1 = checkDate1.split("/");
            checkDate1 = new Date(checkDate1[2], checkDate1[0] - 1, checkDate1[1]).getTime();

            checkDate2 = checkDate2.split("/");
            checkDate2 = new Date(checkDate2[2], checkDate2[0] - 1, checkDate2[1]).getTime();

            if ((checkDate1 - checkDate2) < 0) {
                createRedAlert("Warning", fieldDisplayName1 + " is before " + fieldDisplayName2);
            }
        }
    }
    return true;
}

function createAlert(dialogTitle, dialogContent) {
    $.alert({
        theme: "casas",
        escapeKey: true,
        backgroundDismiss: true,
        useBootstrap: false,
        boxWidth: "80%",
        title: dialogTitle,
        content: dialogContent
    });
}

function createRedAlert(dialogTitle, dialogContent) {
    $.alert({
        theme: "casas-alert",
        escapeKey: true,
        backgroundDismiss: true,
        useBootstrap: false,
        boxWidth: "40%",
        title: dialogTitle,
        content: dialogContent
    });
}

function createRedAlertNoButtons(dialogTitle, dialogContent) {
    $.alert({
        theme: "casas-alert",
        escapeKey: true,
        backgroundDismiss: true,
        useBootstrap: false,
        boxWidth: "40%",
        title: dialogTitle,
        content: dialogContent,
        closeIcon: false, // hides the close icon.
        buttons: {
            button: {
                isHidden: true, // hide the button
                action: function () {
                },

            }
        }
    });
}

function createConfirm(dialogTitle, dialogContent, confirmUrl) {
    $.confirm({
        theme: "casas",
        escapeKey: true,
        backgroundDismiss: true,
        useBootstrap: false,
        boxWidth: "50%",
        title: dialogTitle,
        content: dialogContent,
        buttons: {
            confirm: function () {
                window.location.href = confirmUrl;
            },
            cancel: function () {
                //do nothing
            }
        }
    });
}

function disableBackButton() {
    window.history.forward();
}

/*Used to search table data and return only rows that match*/
function searchColumn(tableId) {

    var inputs, input, filter, table, tr, td, i, x, y, txtValue;
    inputs = document.getElementsByClassName('search-input');
    table = document.getElementById(tableId);

    tr = table.getElementsByTagName("tr");

    //show all rows again to start off
    for (y = 0; y < tr.length; y++) {
        tr[y].style.display = "";
    }

    //loop through each input re-filtering the rows
    for (x = 0; x < inputs.length; x++) {

        //get current filter input value
        input = inputs[x];
        filter = input.value.toUpperCase();

        //loop through the rows, filtering out ones in the filtered column that don't match
        for (i = 0; i < tr.length; i++) {

            //don't look at rows already removed by other filters
            if(tr[i].style.display !== "none") {

                td = tr[i].getElementsByTagName("td")[x];
                if (td) {
                    txtValue = td.textContent || td.innerText;
                    if (txtValue.toUpperCase().indexOf(filter) > -1) {
                        tr[i].style.display = "";
                    } else {
                        tr[i].style.display = "none";
                    }
                }
            }
        }
    }
}

/**
 * Sort table in the page using QuickSort algorithm
 * @param tableId ID of table to sort
 * @param startRow Row to start with, typically 1 to skip the header (row 0)
 * @param columnNumber Column to start, start counting at 0
 * @param direction Direction to sort, asc or desc
 */
function sortTable(tableId, startRow, columnNumber, direction) {
    //find the tbody section of the table to avoid sorting the header rows
    var tbody = $("#" + tableId).find("tbody");

    //find all the rows in the tbody section
    var trs = tbody.find("tr");

    //sort the rows based on the column and direction chosen
    trs.sort(function(a, b){
        var aValue = a.getElementsByTagName("TD")[columnNumber].getElementsByClassName("sortable-content")[0].innerHTML.toLowerCase();
        if(!aValue) aValue = 'zzzzzzz';

        var bValue = b.getElementsByTagName("TD")[columnNumber].getElementsByClassName("sortable-content")[0].innerHTML.toLowerCase();
        if(!bValue) bValue = 'zzzzzzz';

        if (direction === 'asc') {
            return ((aValue < bValue) ? -1 : ((aValue > bValue) ? 1 : 0));
        } else {
            return ((aValue > bValue) ? -1 : ((aValue < bValue) ? 1 : 0));
        }
    });

    //note: at first I tried insert rows above and below other rows while sorting and this
    //was much slower than just wiping out the rows and recreating them in a new order

    //remove the rows
    tbody.empty();

    //add the reordered rows
    $(trs).appendTo(tbody);


}
