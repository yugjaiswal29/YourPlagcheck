document.querySelectorAll("[data-dropzone]").forEach((zone) => {
    const input = zone.querySelector('input[type="file"]');
    const fileList = zone.querySelector("[data-file-list]");
    if (!input || !fileList) return;

    const showFiles = (files) => {
        fileList.replaceChildren();
        if (!files.length) {
            fileList.classList.remove("has-files");
            return;
        }

        const names = Array.from(files).map((file) => file.name);
        const label = document.createElement("span");
        label.textContent = names.join(", ");
        fileList.append(label);
        fileList.classList.add("has-files");
    };

    input.addEventListener("change", () => showFiles(input.files));
    ["dragenter", "dragover"].forEach((eventName) => {
        zone.addEventListener(eventName, (event) => {
            event.preventDefault();
            zone.classList.add("is-dragover");
        });
    });
    ["dragleave", "drop"].forEach((eventName) => {
        zone.addEventListener(eventName, (event) => {
            event.preventDefault();
            zone.classList.remove("is-dragover");
        });
    });
    zone.addEventListener("drop", (event) => {
        const files = event.dataTransfer?.files;
        if (!files?.length) return;

        const transfer = new DataTransfer();
        Array.from(files).forEach((file) => transfer.items.add(file));
        input.files = transfer.files;
        showFiles(input.files);
    });
});
