// Various hints and extra info for the settings tab.

let settingsHintsObserver = null;
let settingsHintsTimer = null;

function settingsHintsLabel(div) {
    // Gradio 5.29 moved block-info outside labels; keep the older fallback.
    return (
        div.querySelector(":scope > .container > span[data-testid='block-info']") ||
        div.querySelector(":scope > label > span[data-testid='block-info']") ||
        div.querySelector(":scope > label > span") ||
        div.querySelector("span[data-testid='block-info']") ||
        div.querySelector("label span")
    );
}

function settingsHintsApply() {
    let root = gradioApp();
    let settingsJson = root.querySelector("#settings_json textarea");
    if (!settingsJson) return;

    let options;
    try {
        options = JSON.parse(settingsJson.value || "{}");
    } catch (e) {
        return;
    }

    if (!("_comments_before" in options) || !("_comments_after" in options)) return;

    let commentsBefore = options._comments_before || {};
    let commentsAfter = options._comments_after || {};

    root.querySelectorAll("#settings [id^=setting_]").forEach(function (div) {
        if (div.dataset.settingsHintsApplied === "1") return;

        let name = div.id.substr(8);
        let commentBefore = commentsBefore[name];
        let commentAfter = commentsAfter[name];

        if (!commentBefore && !commentAfter) {
            div.dataset.settingsHintsApplied = "1";
            return;
        }

        let label = settingsHintsLabel(div);
        if (!label || !label.parentElement) return;

        if (commentBefore) {
            let comment = document.createElement("DIV");
            comment.className = "settings-comment";
            comment.innerHTML = commentBefore;
            label.parentElement.insertBefore(document.createTextNode("\xa0"), label);
            label.parentElement.insertBefore(comment, label);
            label.parentElement.insertBefore(document.createTextNode("\xa0"), label);
        }

        if (commentAfter) {
            let comment = document.createElement("DIV");
            comment.className = "settings-comment";
            comment.innerHTML = commentAfter;
            label.parentElement.insertBefore(comment, label.nextSibling);
            label.parentElement.insertBefore(document.createTextNode("\xa0"), comment.nextSibling);
        }

        div.dataset.settingsHintsApplied = "1";
    });
}

function settingsHintsSchedule() {
    clearTimeout(settingsHintsTimer);
    settingsHintsTimer = setTimeout(settingsHintsApply, 0);
}

function settingsHintsStart() {
    if (settingsHintsObserver) {
        settingsHintsSchedule();
        return;
    }

    let root = gradioApp();
    settingsHintsObserver = new MutationObserver(function () {
        if (root.querySelector("#settings [id^=setting_], #settings_json textarea")) settingsHintsSchedule();
    });
    settingsHintsObserver.observe(root, { childList: true, subtree: true });
    settingsHintsSchedule();
}

onUiLoaded(settingsHintsStart);
onOptionsAvailable(settingsHintsStart);
onOptionsChanged(settingsHintsSchedule);
setTimeout(settingsHintsStart, 0);
setTimeout(settingsHintsStart, 500);
setTimeout(settingsHintsStart, 1500);

function settingsHintsShowQuicksettings() {
    requestGet("./internal/quicksettings-hint", {}, function (data) {
        let table = document.createElement("table");
        table.className = "popup-table";

        data.forEach(function (obj) {
            let tr = document.createElement("tr");
            let td = document.createElement("td");
            td.textContent = obj.name;
            tr.appendChild(td);

            td = document.createElement("td");
            td.textContent = obj.label;
            tr.appendChild(td);

            table.appendChild(tr);
        });

        popup(table);
    });
}

// Gradio 5 resolves component _js callbacks from window.
window.settingsHintsShowQuicksettings = settingsHintsShowQuicksettings;
