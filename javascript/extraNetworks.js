function toggleCss(key, css, enable) {
    let style = document.getElementById(key);
    if (enable && !style) {
        style = document.createElement("style");
        style.id = key;
        style.type = "text/css";
        document.head.appendChild(style);
    }
    if (style && !enable) {
        document.head.removeChild(style);
    }
    if (style) {
        style.innerHTML = "";
        style.appendChild(document.createTextNode(css));
    }
}

function setupExtraNetworksForTab(tabname) {
    function registerPrompt(tabname, id) {
        let textarea = gradioApp().querySelector("#" + id + " textarea[data-testid='textbox']");

        if (!textarea) {
            return;
        }

        if (!activePromptTextarea[tabname]) {
            activePromptTextarea[tabname] = textarea;
        }

        textarea.addEventListener("focus", function () {
            activePromptTextarea[tabname] = textarea;
        });
    }

    let extraTabs = gradioApp().getElementById(tabname + "_extra_tabs");
    if (!extraTabs) {
        return;
    }

    // The role selector survives Gradio 4/5 wrapper changes.
    let tabnav = extraTabs.querySelector(":scope > .tab-wrapper > [role='tablist']");
    if (!tabnav) {
        tabnav = extraTabs.querySelector("[role='tablist']");
    }
    if (!tabnav) {
        return;
    }

    let controlsDiv = tabnav.querySelector(":scope > .extra-networks-controls-div");
    if (!controlsDiv) {
        controlsDiv = document.createElement("DIV");
        controlsDiv.classList.add("extra-networks-controls-div");
        tabnav.appendChild(controlsDiv);
    }

    // Resolve panes by class/id because Gradio nests them differently by version.
    extraTabs.querySelectorAll(".extra-network-pane[id$='_pane']").forEach(function (pane) {
        // tabname_full = {tabname}_{extra_networks_tabname}
        let tabname_full = pane.id.slice(0, -"_pane".length);
        let search = pane.querySelector("#" + tabname_full + "_extra_search");
        let sort_dir = pane.querySelector("#" + tabname_full + "_extra_sort_dir");
        let refresh = pane.querySelector("#" + tabname_full + "_extra_refresh");
        let currentSort = "";

        // If any of the buttons above don't exist, we want to skip this iteration of the loop.
        if (!search || !sort_dir || !refresh) {
            return; // `return` is equivalent of `continue` but for forEach loops.
        }

        let applyFilter = function (force) {
            // Refresh can replace these nodes, so resolve them on each filter.
            let currentSearch = gradioApp().getElementById(tabname_full + "_extra_search");
            let cardsContainer = gradioApp().getElementById(tabname_full + "_cards");
            if (!currentSearch || !cardsContainer) return;

            let searchTerm = (currentSearch.value || "").trim().toLowerCase();
            let splitSearch = searchTerm.split(/\s+/).filter(Boolean);

            let hiddenMode = typeof opts !== "undefined" && opts.extra_networks_hidden_models
                ? opts.extra_networks_hidden_models
                : "Always";

            cardsContainer.querySelectorAll(":scope .card").forEach(function (elem) {
                let hiddenDirectory = elem.classList.contains("extra-network-hidden-directory");
                let searchOnly = elem.classList.contains("search_only") || elem.querySelector(".search_only");
                let text = Array.prototype.map
                    .call(elem.querySelectorAll(".name, .search_terms, .description"), function (t) {
                        return t.textContent.toLowerCase();
                    })
                    .join(" ");

                let visible = true;
                if (hiddenDirectory && hiddenMode === "Never") {
                    visible = false;
                } else if ((searchOnly || (hiddenDirectory && hiddenMode === "When searched")) && searchTerm.length < 4) {
                    visible = false;
                }

                splitSearch.forEach(function (partial) {
                    if (text.indexOf(partial) == -1) visible = false;
                });

                elem.classList.toggle("hidden", !visible);
            });

            applySort(force);
        };

        let applySort = function (force) {
            let currentSortDir = gradioApp().getElementById(tabname_full + "_extra_sort_dir");
            let cardsContainer = gradioApp().getElementById(tabname_full + "_cards");
            if (!currentSortDir || !cardsContainer) return;

            let cardParents = Array.from(
                cardsContainer.querySelectorAll(":scope > .extra-network-category > .extra-network-category-cards"),
            );
            if (cardsContainer.querySelector(":scope > .card")) {
                cardParents.unshift(cardsContainer);
            }
            if (cardParents.length === 0) {
                cardParents = [cardsContainer];
            }

            let cards = cardParents.flatMap(function (parent) {
                return Array.from(parent.querySelectorAll(":scope > .card"));
            });
            let reverse = currentSortDir.dataset.sortdir == "Descending";
            let activeSearchElem = gradioApp().querySelector(
                "#" + tabname_full + "_controls .extra-network-control--sort.extra-network-control--enabled",
            );
            let sortKey = activeSearchElem ? activeSearchElem.dataset.sortkey : "default";
            let sortKeyDataField = "sort" + sortKey.charAt(0).toUpperCase() + sortKey.slice(1);
            let sortKeyStore = sortKey + "-" + currentSortDir.dataset.sortdir + "-" + cards.length + "-" + cardParents.length;

            if (sortKeyStore == currentSort && !force) {
                return;
            }
            currentSort = sortKeyStore;

            cardParents.forEach(function (parent) {
                let sortedCards = Array.from(parent.querySelectorAll(":scope > .card"));
                sortedCards.sort(function (cardA, cardB) {
                    let a = cardA.dataset[sortKeyDataField];
                    let b = cardB.dataset[sortKeyDataField];
                    if (!isNaN(a) && !isNaN(b)) {
                        return parseInt(a) - parseInt(b);
                    }

                    return a < b ? -1 : a > b ? 1 : 0;
                });

                if (reverse) {
                    sortedCards.reverse();
                }

                let frag = document.createDocumentFragment();
                sortedCards.forEach(function (card) {
                    frag.appendChild(card);
                });
                parent.appendChild(frag);
            });
        };

        if (search.dataset.extraNetworksBound !== "true") {
            // `input` fires for every typed character, paste, and clear-button
            // action. Do not wait for Enter or a Gradio backend event.
            search.addEventListener("input", function () {
                applyFilter();
            });
            // The controls live inside Gradio's tablist. Stop the tablist
            // handler from stealing focus when the search field is touched.
            // Capture the pointer/touch events as well: mobile browsers may
            // dispatch pointer events before the compatibility mouse events.
            ["pointerdown", "pointerup", "pointercancel", "mousedown", "mouseup", "click", "touchstart", "touchend", "keydown"].forEach(function (eventName) {
                search.addEventListener(eventName, function (event) {
                    event.stopPropagation();
                }, true);
            });
            search.dataset.extraNetworksBound = "true";
        }
        applySort();
        applyFilter();

        extraNetworksApplySort[tabname_full] = applySort;
        extraNetworksApplyFilter[tabname_full] = applyFilter;

        let controls = pane.querySelector("#" + tabname_full + "_controls");
        if (controls) {
            let oldControls = controlsDiv.querySelector("#" + tabname_full + "_controls");
            if (oldControls && oldControls !== controls) {
                oldControls.remove();
            }
            controlsDiv.appendChild(controls);
        }

        if (pane.offsetParent !== null) {
            extraNetworksShowControlsForPage(tabname, tabname_full);
        }
    });

    extraNetworksNormalizeTabNav(tabname);
    extraNetworksReapplyCategoryPriority(tabname);
    registerPrompt(tabname, tabname + "_prompt");
    registerPrompt(tabname, tabname + "_neg_prompt");
}

function extraNetworksMovePromptToTab(tabname, id, showPrompt, showNegativePrompt) {
    if (!gradioApp().querySelector(".toprow-compact-tools")) return; // only applicable for compact prompt layout

    let promptContainer = gradioApp().getElementById(tabname + "_prompt_container");
    let prompt = gradioApp().getElementById(tabname + "_prompt_row");
    let negPrompt = gradioApp().getElementById(tabname + "_neg_prompt_row");
    let elem = id ? gradioApp().getElementById(id) : null;

    if (showNegativePrompt && elem) {
        elem.insertBefore(negPrompt, elem.firstChild);
    } else {
        promptContainer.insertBefore(negPrompt, promptContainer.firstChild);
    }

    if (showPrompt && elem) {
        elem.insertBefore(prompt, elem.firstChild);
    } else {
        promptContainer.insertBefore(prompt, promptContainer.firstChild);
    }

    if (elem) {
        elem.classList.toggle("extra-page-prompts-active", showNegativePrompt || showPrompt);
    }
}

function extraNetworksNormalizeTabNav(tabname) {
    // Mobile browsers resize the viewport when the keyboard opens. Do not
    // reparent the controls while a search field owns focus: moving its
    // ancestor can blur the field and dismiss the keyboard.
    if (document.activeElement?.matches?.("input[id$='_extra_search']")) {
        return;
    }

    let extraTabs = gradioApp().getElementById(tabname + "_extra_tabs");
    let tabnav = extraTabs?.querySelector(":scope > .tab-wrapper > [role='tablist']") || extraTabs?.querySelector("[role='tablist']");
    let controls = tabnav?.querySelector(":scope > .extra-networks-controls-div");
    if (tabnav && controls) {
        // Keep custom controls after the page tabs after responsive reflow.
        tabnav.appendChild(controls);
    }
}

function extraNetworksShowControlsForPage(tabname, tabname_full) {
    extraNetworksNormalizeTabNav(tabname);
    let controls = gradioApp().querySelector("#" + tabname + "_extra_tabs .extra-networks-controls-div");
    if (!controls) {
        return;
    }

    controls.querySelectorAll(":scope > div").forEach(function (elem) {
        let targetId = tabname_full + "_controls";
        elem.style.display = elem.id == targetId ? "" : "none";
    });
}

function extraNetworksUnrelatedTabSelected(tabname) {
    // called from python when user selects an unrelated tab (generate)
    extraNetworksMovePromptToTab(tabname, "", false, false);

    extraNetworksShowControlsForPage(tabname, null);
}

function extraNetworksTabSelected(tabname, id, showPrompt, showNegativePrompt, tabname_full) {
    // called from python when user selects an extra networks tab
    extraNetworksMovePromptToTab(tabname, id, showPrompt, showNegativePrompt);

    extraNetworksShowControlsForPage(tabname, tabname_full);
    extraNetworksNormalizeTabNav(tabname);
}

function applyExtraNetworkFilter(tabname_full) {
    let doFilter = function () {
        let applyFunction = extraNetworksApplyFilter[tabname_full];

        if (applyFunction) {
            applyFunction(true);
        }
    };
    setTimeout(doFilter, 1);
}

function applyExtraNetworkSort(tabname_full) {
    let doSort = function () {
        let applyFunction = extraNetworksApplySort[tabname_full];
        if (applyFunction) {
            applyFunction(true);
        }
    };
    setTimeout(doSort, 1);
}

let extraNetworksApplyFilter = {};
let extraNetworksApplySort = {};
let activePromptTextarea = {};

window.addEventListener("resize", function () {
    if (document.activeElement?.matches?.("input[id$='_extra_search']")) {
        return;
    }

    extraNetworksNormalizeTabNav("txt2img");
    extraNetworksNormalizeTabNav("img2img");
});

function setupExtraNetworks() {
    setupExtraNetworksForTab("txt2img");
    setupExtraNetworksForTab("img2img");
}

const re_extranet = /<([^:^>]+:[^:]+):[\d.]+>(.*)/s;
const re_extranet_g = /<([^:^>]+:[^:]+):[\d.]+>/g;
const re_extranet_neg = /\(([^:^>]+:[\d.]+)\)/;
const re_extranet_g_neg = /\(([^:^>]+:[\d.]+)\)/g;

function extraNetworksTextSeparator() {
    // Older or partially initialized options payloads may not contain the
    // separator. A LoRA card should still insert one normal leading space.
    return typeof opts !== "undefined" && typeof opts.extra_networks_add_text_separator === "string"
        ? opts.extra_networks_add_text_separator
        : " ";
}

function tryToRemoveExtraNetworkFromPrompt(textarea, text, isNeg) {
    let m = text.match(isNeg ? re_extranet_neg : re_extranet);
    let replaced = false;
    let newTextareaText;
    let extraTextBeforeNet = extraNetworksTextSeparator();
    if (m) {
        let extraTextAfterNet = m[2];
        let partToSearch = m[1];
        let foundAtPosition = -1;
        newTextareaText = textarea.value.replaceAll(isNeg ? re_extranet_g_neg : re_extranet_g, function (found, net, pos) {
            m = found.match(isNeg ? re_extranet_neg : re_extranet);
            if (m[1] == partToSearch) {
                replaced = true;
                foundAtPosition = pos;
                return "";
            }
            return found;
        });
        if (foundAtPosition >= 0) {
            if (extraTextAfterNet && newTextareaText.substr(foundAtPosition, extraTextAfterNet.length) == extraTextAfterNet) {
                newTextareaText =
                    newTextareaText.substr(0, foundAtPosition) +
                    newTextareaText.substr(foundAtPosition + extraTextAfterNet.length);
            }
            if (
                newTextareaText.substr(foundAtPosition - extraTextBeforeNet.length, extraTextBeforeNet.length) ==
                extraTextBeforeNet
            ) {
                newTextareaText =
                    newTextareaText.substr(0, foundAtPosition - extraTextBeforeNet.length) +
                    newTextareaText.substr(foundAtPosition);
            }
        }
    } else {
        newTextareaText = textarea.value.replaceAll(new RegExp(`((?:${extraTextBeforeNet})?${text})`, "g"), "");
        replaced = newTextareaText != textarea.value;
    }

    if (replaced) {
        textarea.value = newTextareaText;
        return true;
    }

    return false;
}

function updatePromptArea(text, textArea, isNeg) {
    if (!tryToRemoveExtraNetworkFromPrompt(textArea, text, isNeg)) {
        textArea.value = textArea.value + extraNetworksTextSeparator() + text;
    }

    updateInput(textArea);
}

function cardClicked(tabname, textToAdd, textToAddNegative, allowNegativePrompt) {
    if (textToAddNegative.length > 0) {
        updatePromptArea(textToAdd, gradioApp().querySelector("#" + tabname + "_prompt textarea[data-testid='textbox']"));
        updatePromptArea(
            textToAddNegative,
            gradioApp().querySelector("#" + tabname + "_neg_prompt textarea[data-testid='textbox']"),
            true,
        );
    } else {
        let textarea = allowNegativePrompt
            ? activePromptTextarea[tabname]
            : gradioApp().querySelector("#" + tabname + "_prompt textarea[data-testid='textbox']");
        updatePromptArea(textToAdd, textarea);
    }
}

function saveCardPreview(event, tabname, filename) {
    let textarea = gradioApp().querySelector("#" + tabname + "_preview_filename textarea[data-testid='textbox']");
    let button = gradioApp().getElementById(tabname + "_save_preview");

    textarea.value = filename;
    updateInput(textarea);

    button.click();

    event.stopPropagation();
    event.preventDefault();
}

function extraNetworksSearchButton(tabname, extra_networks_tabname, event) {
    let searchTextarea = gradioApp().querySelector("#" + tabname + "_" + extra_networks_tabname + "_extra_search");
    if (!searchTextarea) {
        return;
    }
    let button = event.target;
    let text = button.classList.contains("search-all") ? "" : button.textContent.trim();

    searchTextarea.value = text;
    searchTextarea.dispatchEvent(new Event("input", { bubbles: true }));
    updateInput(searchTextarea);
}

function extraNetworksTreeProcessFileClick(event, btn, tabname, extra_networks_tabname) {
    /**
     * Processes `onclick` events when user clicks on files in tree.
     *
     * @param event                     The generated event.
     * @param btn                       The clicked `tree-list-item` button.
     * @param tabname                   The name of the active tab in the sd webui. Ex: txt2img, img2img, etc.
     * @param extra_networks_tabname    The id of the active extraNetworks tab. Ex: lora, checkpoints, etc.
     */
    // NOTE: Currently unused.
    return;
}

function extraNetworksTreeProcessDirectoryClick(event, btn, tabname, extra_networks_tabname) {
    /**
     * Processes `onclick` events when user clicks on directories in tree.
     *
     * Here is how the tree reacts to clicks for various states:
     * unselected unopened directory: Directory is selected and expanded.
     * unselected opened directory: Directory is selected.
     * selected opened directory: Directory is collapsed and deselected.
     * chevron is clicked: Directory is expanded or collapsed. Selected state unchanged.
     *
     * @param event                     The generated event.
     * @param btn                       The clicked `tree-list-item` button.
     * @param tabname                   The name of the active tab in the sd webui. Ex: txt2img, img2img, etc.
     * @param extra_networks_tabname    The id of the active extraNetworks tab. Ex: lora, checkpoints, etc.
     */
    let ul = btn.nextElementSibling;
    // This is the actual target that the user clicked on within the target button.
    // We use this to detect if the chevron was clicked.
    let true_targ = event.target;

    function _expand_or_collapse(_ul, _btn) {
        // Expands <ul> if it is collapsed, collapses otherwise. Updates button attributes.
        if (_ul.hasAttribute("hidden")) {
            _ul.removeAttribute("hidden");
            _btn.dataset.expanded = "";
        } else {
            _ul.setAttribute("hidden", "");
            delete _btn.dataset.expanded;
        }
    }

    function _remove_selected_from_all() {
        // Removes the `selected` attribute from all buttons.
        let sels = document.querySelectorAll("div.tree-list-content");
        [...sels].forEach((el) => {
            delete el.dataset.selected;
        });
    }

    function _select_button(_btn) {
        // Removes `data-selected` attribute from all buttons then adds to passed button.
        _remove_selected_from_all();
        _btn.dataset.selected = "";
    }

    function _update_search(_tabname, _extra_networks_tabname, _search_text) {
        // Update search input with select button's path.
        let search_input_elem = gradioApp().querySelector("#" + tabname + "_" + extra_networks_tabname + "_extra_search");
        if (!search_input_elem) {
            return;
        }
        search_input_elem.value = _search_text;
        search_input_elem.dispatchEvent(new Event("input", { bubbles: true }));
        updateInput(search_input_elem);
    }

    // If user clicks on the chevron, then we do not select the folder.
    if (true_targ.matches(".tree-list-item-action--leading, .tree-list-item-action-chevron")) {
        _expand_or_collapse(ul, btn);
    } else {
        // User clicked anywhere else on the button.
        if ("selected" in btn.dataset && !ul.hasAttribute("hidden")) {
            // If folder is select and open, collapse and deselect button.
            _expand_or_collapse(ul, btn);
            delete btn.dataset.selected;
            _update_search(tabname, extra_networks_tabname, "");
        } else if (!(!("selected" in btn.dataset) && !ul.hasAttribute("hidden"))) {
            // If folder is open and not selected, then we don't collapse; just select.
            // NOTE: Double inversion sucks but it is the clearest way to show the branching here.
            _expand_or_collapse(ul, btn);
            _select_button(btn, tabname, extra_networks_tabname);
            _update_search(tabname, extra_networks_tabname, btn.dataset.path);
        } else {
            // All other cases, just select the button.
            _select_button(btn, tabname, extra_networks_tabname);
            _update_search(tabname, extra_networks_tabname, btn.dataset.path);
        }
    }
}

function extraNetworksTreeOnClick(event, tabname, extra_networks_tabname) {
    /**
     * Handles `onclick` events for buttons within an `extra-network-tree .tree-list--tree`.
     *
     * Determines whether the clicked button in the tree is for a file entry or a directory
     * then calls the appropriate function.
     *
     * @param event                     The generated event.
     * @param tabname                   The name of the active tab in the sd webui. Ex: txt2img, img2img, etc.
     * @param extra_networks_tabname    The id of the active extraNetworks tab. Ex: lora, checkpoints, etc.
     */
    let btn = event.currentTarget;
    let par = btn.parentElement;
    if (par.dataset.treeEntryType === "file") {
        extraNetworksTreeProcessFileClick(event, btn, tabname, extra_networks_tabname);
    } else {
        extraNetworksTreeProcessDirectoryClick(event, btn, tabname, extra_networks_tabname);
    }
}

function extraNetworksControlSortOnClick(event, tabname, extra_networks_tabname) {
    /** Handles `onclick` events for Sort Mode buttons. */

    let self = event.currentTarget;
    let parent = event.currentTarget.parentElement;

    parent.querySelectorAll(".extra-network-control--sort").forEach(function (x) {
        x.classList.remove("extra-network-control--enabled");
    });

    self.classList.add("extra-network-control--enabled");

    applyExtraNetworkSort(tabname + "_" + extra_networks_tabname);
}

function extraNetworksControlSortDirOnClick(event, tabname, extra_networks_tabname) {
    /**
     * Handles `onclick` events for the Sort Direction button.
     *
     * Modifies the data attributes of the Sort Direction button to cycle between
     * ascending and descending sort directions.
     *
     * @param event                     The generated event.
     * @param tabname                   The name of the active tab in the sd webui. Ex: txt2img, img2img, etc.
     * @param extra_networks_tabname    The id of the active extraNetworks tab. Ex: lora, checkpoints, etc.
     */
    if (event.currentTarget.dataset.sortdir == "Ascending") {
        event.currentTarget.dataset.sortdir = "Descending";
        event.currentTarget.setAttribute("title", "Sort descending");
    } else {
        event.currentTarget.dataset.sortdir = "Ascending";
        event.currentTarget.setAttribute("title", "Sort ascending");
    }
    applyExtraNetworkSort(tabname + "_" + extra_networks_tabname);
}

function extraNetworksControlTreeViewOnClick(event, tabname, extra_networks_tabname) {
    /**
     * Handles `onclick` events for the Tree View button.
     *
     * Toggles the tree view in the extra networks pane.
     *
     * @param event                     The generated event.
     * @param tabname                   The name of the active tab in the sd webui. Ex: txt2img, img2img, etc.
     * @param extra_networks_tabname    The id of the active extraNetworks tab. Ex: lora, checkpoints, etc.
     */
    let button = event.currentTarget;
    button.classList.toggle("extra-network-control--enabled");
    let show = !button.classList.contains("extra-network-control--enabled");

    let pane = gradioApp().getElementById(tabname + "_" + extra_networks_tabname + "_pane");
    if (pane) {
        pane.classList.toggle("extra-network-dirs-hidden", show);
    }
}

const extraNetworksDefaultCategoryOrder = ["sd", "sdxl", "flux", "klein", "qwen", "lumina", "zit", "wan", "anima", "ernie", "pid", "krea", "unknown"];
const extraNetworksRelatedCategories = {
    sd: ["sd"],
    xl: ["sdxl"],
    flux: ["flux", "klein"],
    klein: ["klein", "flux"],
    qwen: ["qwen"],
    lumina: ["lumina"],
    zit: ["zit"],
    wan: ["wan"],
    anima: ["anima"],
    ernie: ["ernie"],
    pid: ["pid"],
    krea: ["krea"],
};

function extraNetworksCurrentPreset() {
    const presetInput = gradioApp().querySelector("#forge_ui_preset input");
    return (presetInput?.value || "").toLowerCase();
}

function extraNetworksCategoryPriorityOrder() {
    const preferred = [...(extraNetworksRelatedCategories[extraNetworksCurrentPreset()] || []), "unknown"];
    return [...preferred, ...extraNetworksDefaultCategoryOrder.filter((category) => !preferred.includes(category))];
}

function extraNetworksSetCategoryPriority(tabname, extra_networks_tabname, enabled) {
    let pane = gradioApp().getElementById(tabname + "_" + extra_networks_tabname + "_pane");
    let cardsContainer = pane?.querySelector(".extra-network-cards");
    let button = gradioApp().getElementById(tabname + "_" + extra_networks_tabname + "_extra_group_priority");
    if (!cardsContainer) return;

    let groups = Array.from(cardsContainer.querySelectorAll(":scope > .extra-network-category"));
    const priorityOrder = extraNetworksCategoryPriorityOrder();
    groups.sort(function (a, b) {
        let categoryA = (a.dataset.category || "").toLowerCase();
        let categoryB = (b.dataset.category || "").toLowerCase();
        let orderA = enabled ? priorityOrder.indexOf(categoryA) : Number(a.dataset.categoryOrder);
        let orderB = enabled ? priorityOrder.indexOf(categoryB) : Number(b.dataset.categoryOrder);
        if (orderA < 0) orderA = priorityOrder.length;
        if (orderB < 0) orderB = priorityOrder.length;
        let orderDifference = orderA - orderB;
        return orderDifference || a.dataset.category.localeCompare(b.dataset.category);
    });

    let fragment = document.createDocumentFragment();
    groups.forEach(function (group) {
        fragment.appendChild(group);
    });
    cardsContainer.appendChild(fragment);

    if (button) {
        button.classList.toggle("extra-network-control--enabled", enabled);
        button.title = enabled ? "Prioritize matching groups" : "Use default group order";
    }
}

function extraNetworksReapplyCategoryPriority(tabname) {
    const root = gradioApp();
    root.querySelectorAll(".extra-network-pane[id$='_pane']").forEach(function (pane) {
        const tabnameFull = pane.id.slice(0, -"_pane".length);
        if (!tabnameFull.startsWith(tabname + "_")) return;
        const extraPage = tabnameFull.slice((tabname + "_").length);
        const priorityButton = root.getElementById(tabnameFull + "_extra_group_priority");
        if (priorityButton) {
            extraNetworksSetCategoryPriority(
                tabname,
                extraPage,
                priorityButton.classList.contains("extra-network-control--enabled"),
            );
        }
    });
}

function extraNetworksControlGroupPriorityOnClick(event, tabname, extra_networks_tabname) {
    let button = event.currentTarget;
    let enabled = !button.classList.contains("extra-network-control--enabled");
    extraNetworksSetCategoryPriority(tabname, extra_networks_tabname, enabled);
    event.stopPropagation();
}

function refreshExtraNetworkCategoryOpenState() {
    // Reapply the server state after a preset replaces hidden panes.
    const firstCategory = gradioApp().querySelector("details.extra-network-category");
    if (firstCategory && firstCategory.dataset.autoOpenRelated === "false") {
        // Preserve the current state when auto-open is disabled.
        return;
    }

    const openCategories = new Set(extraNetworksRelatedCategories[extraNetworksCurrentPreset()] || []);
    openCategories.add("unknown");

    gradioApp().querySelectorAll("details.extra-network-category").forEach((group) => {
        const category = (group.dataset.category || "").toLowerCase();
        group.open = openCategories.has(category);
    });
}

function clickLoraRefresh() {
    const targets = [
        "txt2img_lora",
        "txt2img_checkpoints",
        "txt2img_textural_inversion",
        "img2img_lora",
        "img2img_checkpoints",
        "img2img_textural_inversion",
    ];
    targets.forEach(function (t) {
        const tab = gradioApp().getElementById(t + "-button");
        if (tab && tab.getAttribute("aria-selected") == "true") {
            const applyFunction = extraNetworksApplyFilter[t];
            if (applyFunction) {
                applyFunction(true);
            }
        }
    });
    extraNetworksReapplyCategoryPriority("txt2img");
    extraNetworksReapplyCategoryPriority("img2img");
    refreshExtraNetworkCategoryOpenState();
}

function extraNetworksControlRefreshOnClick(event, tabname, extra_networks_tabname) {
    /**
     * Handles `onclick` events for the Refresh Page button.
     *
     * In order to actually call the python functions in `ui_extra_networks.py`
     * to refresh the page, we created an empty gradio button in that file with an
     * event handler that refreshes the page. So what this function here does
     * is it manually raises a `click` event on that button.
     *
     * @param event                     The generated event.
     * @param tabname                   The name of the active tab in the sd webui. Ex: txt2img, img2img, etc.
     * @param extra_networks_tabname    The id of the active extraNetworks tab. Ex: lora, checkpoints, etc.
     */
    event?.preventDefault();
    event?.stopPropagation();

    let refresh_component = extraNetworksFindElementById(
        tabname + "_" + extra_networks_tabname + "_extra_refresh_internal",
    );
    let btn_refresh_internal = refresh_component?.matches("button")
        ? refresh_component
        : refresh_component?.querySelector("button") || refresh_component;
    if (btn_refresh_internal && typeof btn_refresh_internal.click === "function") {
        // Use a real bubbling click so Gradio's delegated handler sees it.
        btn_refresh_internal.click();
    }
}

let globalPopup = null;
let globalPopupInner = null;

function extraNetworksFindElementById(id) {
    // Gradio may move the editor between its root and document.body.
    return document.getElementById(id) || gradioApp().getElementById(id);
}

function extraNetworksFindElement(selector) {
    // The textarea and trigger can move with the editor too.
    return document.querySelector(selector) || gradioApp().querySelector(selector);
}

function closePopup() {
    if (!globalPopup) return;
    globalPopup.style.display = "none";
}

function popup(contents) {
    if (!contents) {
        return;
    }

    if (!globalPopup) {
        globalPopup = document.createElement("div");
        globalPopup.classList.add("global-popup");

        let close = document.createElement("div");
        close.classList.add("global-popup-close");
        close.addEventListener("click", closePopup);
        close.title = "Close";
        globalPopup.appendChild(close);

        globalPopupInner = document.createElement("div");
        globalPopupInner.classList.add("global-popup-inner");
        globalPopup.appendChild(globalPopupInner);

        // Gradio 4 exposed `.main`; Gradio 5 no longer does. The popup is
        // fixed-positioned, so attach it to the document body in both
        // layouts instead of returning with an unattached dialog.
        document.body.appendChild(globalPopup);
    }

    // Do not detach the same Gradio editor from its popup on every open.
    // Detaching the live component invalidates Gradio's event/visibility
    // bookkeeping and makes the editor disappear after the next selection.
    if (globalPopupInner.firstElementChild !== contents) {
        globalPopupInner.replaceChildren(contents);
    }

    globalPopup.style.display = "flex";
}

function popupId(id) {
    // Gradio may replace a dialog root after it is opened. Resolve it every
    // time instead of keeping a stale DOM reference from the first opening.
    popup(extraNetworksFindElementById(id));
}

function extraNetworksFlattenMetadata(obj) {
    const result = {};

    // Convert any stringified JSON objects to actual objects
    for (const key of Object.keys(obj)) {
        if (typeof obj[key] === "string") {
            try {
                const parsed = JSON.parse(obj[key]);
                if (parsed && typeof parsed === "object") {
                    obj[key] = parsed;
                }
            } catch (error) {
                continue;
            }
        }
    }

    // Flatten the object
    for (const key of Object.keys(obj)) {
        if (typeof obj[key] === "object" && obj[key] !== null) {
            const nested = extraNetworksFlattenMetadata(obj[key]);
            for (const nestedKey of Object.keys(nested)) {
                result[`${key}/${nestedKey}`] = nested[nestedKey];
            }
        } else {
            result[key] = obj[key];
        }
    }

    // Special case for handling modelspec keys
    for (const key of Object.keys(result)) {
        if (key.startsWith("modelspec.")) {
            result[key.replaceAll(".", "/")] = result[key];
            delete result[key];
        }
    }

    // Add empty keys to designate hierarchy
    for (const key of Object.keys(result)) {
        const parts = key.split("/");
        for (let i = 1; i < parts.length; i++) {
            const parent = parts.slice(0, i).join("/");
            if (!result[parent]) {
                result[parent] = "";
            }
        }
    }

    return result;
}

function extraNetworksShowMetadata(text) {
    try {
        let parsed = JSON.parse(text);
        if (parsed && typeof parsed === "object") {
            parsed = extraNetworksFlattenMetadata(parsed);
            const table = createVisualizationTable(parsed, 0);
            popup(table);
            return;
        }
    } catch (error) {
        console.error(error);
    }

    let elem = document.createElement("pre");
    elem.classList.add("popup-metadata");
    elem.textContent = text;

    popup(elem);
    return;
}

function requestGet(url, data, handler, errorHandler) {
    let xhr = new XMLHttpRequest();
    let args = Object.keys(data)
        .map(function (k) {
            return encodeURIComponent(k) + "=" + encodeURIComponent(data[k]);
        })
        .join("&");
    xhr.open("GET", url + "?" + args, true);

    xhr.onreadystatechange = function () {
        if (xhr.readyState === 4) {
            if (xhr.status === 200) {
                try {
                    let js = JSON.parse(xhr.responseText);
                    handler(js);
                } catch (error) {
                    console.error(error);
                    errorHandler();
                }
            } else {
                errorHandler();
            }
        }
    };
    let js = JSON.stringify(data);
    xhr.send(js);
}

function extraNetworksCopyCardPath(event) {
    navigator.clipboard.writeText(event.target.getAttribute("data-clipboard-text"));
    event.stopPropagation();
}

function extraNetworksRequestMetadata(event, extraPage) {
    let showError = function () {
        extraNetworksShowMetadata("there was an error getting metadata");
    };

    let card = event.currentTarget.closest(".card");
    let cardName = card ? card.getAttribute("data-name") : null;
    if (cardName == null) {
        // Tree-view metadata actions have one extra wrapper.
        card = event.currentTarget.closest("[data-name]");
        cardName = card ? card.getAttribute("data-name") : null;
    }
    if (cardName == null) {
        showError();
        event.stopPropagation();
        return;
    }

    requestGet(
        "./sd_extra_networks/metadata",
        { page: extraPage, item: cardName },
        function (data) {
            if (data && data.metadata) {
                extraNetworksShowMetadata(data.metadata);
            } else {
                showError();
            }
        },
        showError,
    );

    event.stopPropagation();
}

let extraPageUserMetadataEditors = {};

function extraNetworksEditUserMetadata(event, tabname, extraPage) {
    let id = tabname + "_" + extraPage + "_edit_user_metadata";

    let editor = extraPageUserMetadataEditors[id] || {};
    // Re-resolve after Gradio replaces the editor, but retain detached nodes.
    editor.page = extraNetworksFindElementById(id) || editor.page;
    editor.nameTextarea = extraNetworksFindElement("#" + id + "_name textarea") || editor.nameTextarea;
    editor.button = extraNetworksFindElement("#" + id + "_button") || editor.button;
    extraPageUserMetadataEditors[id] = editor;

    if (!editor.page || !editor.nameTextarea || !editor.button) {
        event.stopPropagation();
        return;
    }

    let card = event.currentTarget.closest(".card") || event.currentTarget.closest("[data-name]");
    let cardName = card ? card.getAttribute("data-name") : null;
    if (cardName == null) {
        event.stopPropagation();
        return;
    }
    editor.nameTextarea.value = cardName;
    updateInput(editor.nameTextarea);

    // Attach the live editor before raising its hidden Gradio event.
    popup(editor.page);
    editor.button.click();

    event.stopPropagation();
}

function extraNetworksRefreshSingleCard(page, tabname, name) {
    requestGet("./sd_extra_networks/get-single-card", { page: page, tabname: tabname, name: name }, function (data) {
        if (data && data.html) {
            let cardsContainer = gradioApp().getElementById(`${tabname}_${page.replace(" ", "_")}_cards`);
            let card = cardsContainer
                ? Array.from(cardsContainer.querySelectorAll(":scope .card")).find((candidate) => candidate.dataset.name === name)
                : null;
            if (!card) {
                return;
            }

            let newDiv = document.createElement("DIV");
            newDiv.innerHTML = data.html;
            let newCard = newDiv.firstElementChild;

            newCard.style.display = "";
            let targetParent = card.parentElement;
            let category = newCard.dataset.category;
            if (category && cardsContainer.querySelector(":scope > .extra-network-category")) {
                let categoryGroup = Array.from(
                    cardsContainer.querySelectorAll(":scope > .extra-network-category"),
                ).find((candidate) => candidate.dataset.category === category);

                if (!categoryGroup) {
                    categoryGroup = document.createElement("details");
                    categoryGroup.className = "extra-network-category";
                    categoryGroup.dataset.category = category;
                    categoryGroup.open = true;
                    let summary = document.createElement("summary");
                    summary.textContent = category;
                    categoryGroup.appendChild(summary);
                    let categoryCards = document.createElement("div");
                    categoryCards.className = "extra-network-category-cards";
                    categoryCards.dataset.category = category;
                    categoryGroup.appendChild(categoryCards);
                    cardsContainer.appendChild(categoryGroup);
                }

                targetParent = categoryGroup.querySelector(":scope > .extra-network-category-cards");
            }

            targetParent.insertBefore(newCard, targetParent === card.parentElement ? card : null);
            card.remove();
            let applyFunction = extraNetworksApplyFilter[`${tabname}_${page.replace(" ", "_")}`];
            if (applyFunction) {
                applyFunction(true);
            }
        }
    });
}

window.addEventListener("keydown", function (event) {
    if (event.key == "Escape") {
        closePopup();
    }
});

/**
 * Setup custom loading for this script.
 * We need to wait for all of our HTML to be generated in the extra networks tabs
 * before we can actually run the `setupExtraNetworks` function.
 * The `onUiLoaded` function actually runs before all of our extra network tabs are
 * finished generating. Thus we needed this new method.
 *
 */

let uiAfterScriptsCallbacks = [];
let uiAfterScriptsTimeout = null;
let executedAfterScripts = false;

function scheduleAfterScriptsCallbacks() {
    clearTimeout(uiAfterScriptsTimeout);
    uiAfterScriptsTimeout = setTimeout(function () {
        executeCallbacks(uiAfterScriptsCallbacks);
    }, 200);
}

onUiLoaded(function () {
    let mutationObserver = new MutationObserver(function (m) {
        let existingSearchfields = gradioApp().querySelectorAll("[id$='_extra_search']").length;
        let tabButtons = gradioApp().querySelectorAll("[id$='_extra_tabs'] [role='tab']").length;
        let neededSearchfields = tabButtons > 0 ? Math.max(1, tabButtons - 2) : existingSearchfields;

        if (!executedAfterScripts && existingSearchfields >= neededSearchfields) {
            mutationObserver.disconnect();
            executedAfterScripts = true;
            scheduleAfterScriptsCallbacks();
        }
    });
    mutationObserver.observe(gradioApp(), { childList: true, subtree: true });
});

uiAfterScriptsCallbacks.push(setupExtraNetworks);
