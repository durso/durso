<?php
/**
 * Description of form
 *
 * @author durso
 */
namespace library\dom\elements\components;
use library\dom\elements\paired;

class form extends paired{
    
    public function __construct($action, $method = "POST") {
        parent::__construct();
        $this->attributes["action"] = $action;
        $this->attributes["method"] = $method;
        $this->tag = "form";
        $this->setId("form");
    }
    
}
