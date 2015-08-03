<?php
/**
 * Description of head
 *
 * @author durso
 */
namespace library\dom\elements\components;
use library\dom\elements\paired;

class head extends paired {
    
    public function __construct() {
        parent::__construct();
        $this->tag = "head";
    }
}
