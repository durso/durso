<?php
/**
 * Description of form
 *
 * @author durso
 */
namespace library\layout\elements;
use library\layout\elements\element;

class form extends element{
    
    public function __construct($action, $method = "POST") {
        $this->attributes["action"] = $action;
        $this->attributes["method"] = $method;
        $this->tag = "form";
        $this->closeTag = true;
    }
    
}
