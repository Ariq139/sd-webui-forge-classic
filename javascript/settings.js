let settingsExcludeTabsFromShowAll = {
    settings_tab_defaults: 1,
    settings_tab_sysinfo: 1,
    settings_tab_actions: 1,
    settings_tab_licenses: 1,
};

let settingsSearchTimer = null;
let settingsSelectedPanel = null;

function settingsPanels() {
    return Array.from(gradioApp().querySelectorAll("#settings > .tabitem[id^=settings_]")
    );
}

function settingsMetadata() {
    let textarea = gradioApp().querySelector("#settings_nav_data textarea");

    try {
        let value = JSON.parse(textarea?.value || "[]");
        return Array.isArray(value) ? value : [];
    } catch (e) {
        return [];
    }
}

function settingsSidebar() {
    let settings = gradioApp().querySelector("#settings");
    let wrapper = settings?.querySelector(":scope > .tab-wrapper");
    if (!wrapper) return null;

    let sidebar = wrapper.querySelector(":scope > .settings-sidebar");
    if (!sidebar) {
        sidebar = document.createElement("div");
        sidebar.className = "settings-sidebar";
        wrapper.appendChild(sidebar);
    }

    let search = gradioApp().querySelector("#settings_search");
    let showAll = gradioApp().querySelector("#settings_show_all_pages");

    if (search && search.parentElement !== sidebar) {
        sidebar.insertBefore(search, sidebar.firstChild);
    }

    if (showAll && showAll.parentElement !== sidebar) {
        sidebar.appendChild(showAll);
    }

    return sidebar;
}

function settingsSelectPanel(panelId, selectedButton) {
    let panels = settingsPanels();
    let panel = panels.find((elem) => elem.id === panelId);
    if (!panel) return false;

    panels.forEach((elem) => {
        elem.style.display = elem === panel ? "flex" : "none";
    });

    settingsSelectedPanel = panelId;
    gradioApp().querySelectorAll(".settings-nav-button").forEach((button) => {
        button.classList.toggle("selected", button === selectedButton || button.dataset.settingsPanel === panelId);
    });

    if (panelId === "settings_tab_licenses" && typeof populateLicense === "function") {
        populateLicense();
    }

    return true;
}

function settingsShowAllTabs() {
    settingsPanels().forEach(function (elem) {
        elem.style.display = settingsExcludeTabsFromShowAll[elem.id] ? "none" : "flex";
    });

    settingsSelectedPanel = null;
    gradioApp().querySelectorAll(".settings-nav-button").forEach((button) => button.classList.remove("selected"));
}

function settingsShowOneTab() {
    let activePanel = settingsSelectedPanel || settingsPanels().find((elem) => getComputedStyle(elem).display !== "none")?.id;
    let button = activePanel && gradioApp().querySelector(`[data-settings-panel="${activePanel}"]`);

    if (!activePanel) {
        activePanel = settingsMetadata()[0]?.id;
        button = activePanel && gradioApp().querySelector(`[data-settings-panel="${activePanel}"]`);
    }

    if (activePanel) settingsSelectPanel(activePanel, button);
}

function setupSettingsControls() {
    return settingsSidebar();
}

function settingsSetupNavigation() {
    let sidebar = settingsSidebar();
    if (!sidebar) return null;

    let metadata = settingsMetadata();
    if (!metadata.length || sidebar.dataset.settingsNavBuilt === "1") return sidebar;

    metadata.forEach((item) => {
        if (!item?.id || !item?.label || !settingsPanels().some((panel) => panel.id === item.id)) return;

        let button = document.createElement("button");
        button.type = "button";
        button.className = "settings-nav-button secondary";
        button.dataset.settingsPanel = item.id;
        button.textContent = item.label;
        button.addEventListener("click", () => settingsSelectPanel(item.id, button));

        let showAll = gradioApp().querySelector("#settings_show_all_pages");
        if (showAll && showAll.parentElement === sidebar) sidebar.insertBefore(button, showAll);
        else sidebar.appendChild(button);
    });

    let initialPanel = settingsPanels().find((elem) => getComputedStyle(elem).display !== "none")?.id || metadata[0]?.id;
    if (initialPanel) {
        let initialButton = sidebar.querySelector(`[data-settings-panel="${initialPanel}"]`);
        settingsSelectPanel(initialPanel, initialButton);
    }

    sidebar.dataset.settingsNavBuilt = "1";
    return sidebar;
}

function addSettingsCategories() {
    let settingsSidebarElement = settingsSetupNavigation();
    if (!settingsSidebarElement || settingsSidebarElement.querySelector(".settings-category")) return;

    let settingsJson = gradioApp().querySelector("#settings_json textarea");
    let categories = [];
    try {
        categories = JSON.parse(settingsJson?.value || "{}")._categories || [];
    } catch (e) {
        return;
    }

    if (!Array.isArray(categories)) categories = [];

    let sectionMap = {};
    settingsSidebarElement.querySelectorAll(":scope > .settings-nav-button").forEach(function (button) {
        let text = button.textContent.trim();
        if (text) sectionMap[text] = button;
    });

    categories.forEach(function (x) {
        let section = localization[x[0]] ?? x[0];
        let category = localization[x[1]] ?? x[1];

        if (!section || !category || category === "undefined") return;

        let sectionElem = sectionMap[section];
        if (!sectionElem) return;

        let span = document.createElement("SPAN");
        span.textContent = category;
        span.className = "settings-category";
        settingsSidebarElement.insertBefore(span, sectionElem);
    });
}

function setupSettingsSearch() {
    let settingsSidebarElement = settingsSetupNavigation();
    let edit = gradioApp().querySelector("#settings_search");
    let editTextarea = gradioApp().querySelector("#settings_search input[data-testid='textbox']");
    let buttonShowAllPages = gradioApp().getElementById("settings_show_all_pages");

    if (!edit || !editTextarea || !buttonShowAllPages || !settingsSidebarElement) return false;

    if (editTextarea.dataset.sdWebuiSettingsSearchBound !== "1") {
        editTextarea.addEventListener("input", function () {
            clearTimeout(settingsSearchTimer);
            settingsSearchTimer = setTimeout(function () {
                let searchText = (editTextarea.value || "").trim().toLowerCase();

                settingsPanels().forEach((panel) => {
                    let controls = panel.querySelectorAll("[id^=column_settings_] > *");
                    controls.forEach(function (elem) {
                        let visible = !searchText || elem.textContent.trim().toLowerCase().indexOf(searchText) !== -1;
                        elem.style.display = visible ? "" : "none";
                    });
                });

                if (searchText !== "") settingsShowAllTabs();
                else settingsShowOneTab();
            }, 500);
        });

        editTextarea.dataset.sdWebuiSettingsSearchBound = "1";
    }

    if (buttonShowAllPages.dataset.sdWebuiSettingsShowAllBound !== "1") {
        buttonShowAllPages.addEventListener("click", settingsShowAllTabs);
        buttonShowAllPages.dataset.sdWebuiSettingsShowAllBound = "1";
    }

    return true;
}

function startSettingsLayoutObserver() {
    let root = gradioApp();

    function initializeSettings() {
        setupSettingsControls();
        settingsSetupNavigation();
        setupSettingsSearch();
        addSettingsCategories();
    }

    let observer = new MutationObserver(initializeSettings);
    observer.observe(root, { childList: true, subtree: true });
    initializeSettings();
    setTimeout(initializeSettings, 0);
    setTimeout(initializeSettings, 500);
    setTimeout(initializeSettings, 1500);
}

if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", startSettingsLayoutObserver);
} else {
    startSettingsLayoutObserver();
}

onOptionsChanged(function () {
    settingsSetupNavigation();
    addSettingsCategories();
});

onOptionsAvailable(function () {
    settingsSetupNavigation();
    addSettingsCategories();
});
